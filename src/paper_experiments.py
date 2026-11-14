"""Exact implementation of Brownlee et al. (GI 2021) Experiments 1, 2, and 3."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier

from src.config import (
    CV_FOLDS,
    INFERENCE_REPEATS,
    MORTGAGE_MLP_GRID,
    NSGA2_CONFIG,
    RANDOM_SEED,
    RESULTS_DIR,
    UCI_MLP_GRID,
    expand_mlp_grid,
)
from src.data_loader import PreparedDataset, load_dataset
from src.energy_profiler import EnergyProfiler
from src.pareto import (
    compute_attainment_curves,
    compute_hypervolume_2d,
    extract_pareto_front,
    find_knee_point,
)


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    avg = "binary" if len(np.unique(y_true)) == 2 else "weighted"
    return {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
    }


def _iqr(arr: np.ndarray) -> float:
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))


def _fit_and_measure_once(
    ds: PreparedDataset,
    cfg: Dict[str, Any],
    profiler: EnergyProfiler,
    inference_repeats: int,
    seed: int,
) -> Dict[str, Any]:
    """Run one training fit + repeated inference measurement for an MLP configuration."""
    model = MLPClassifier(
        hidden_layer_sizes=cfg["hidden_layer_sizes"],
        activation=cfg["activation"],
        solver=cfg["solver"],
        alpha=float(cfg["alpha"]),
        max_iter=int(cfg["max_iter"]),
        random_state=seed,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)

        _, train_meas = profiler.measure_callable(lambda: model.fit(ds.X_train, ds.y_train))

        def _run_inference() -> np.ndarray:
            preds = None
            for _ in range(inference_repeats):
                preds = model.predict(ds.X_test)
            return preds

        y_pred, inf_meas = profiler.measure_callable(_run_inference)

    metrics = _classification_metrics(ds.y_test, y_pred)
    # Normalize inference energy to 1,000 test applications (Paper Section IV standard)
    inf_scale = 1000.0 / max(1, inference_repeats)

    return {
        "train_energy_j": train_meas.cpu_energy_j,
        "train_total_energy_j": train_meas.total_energy_j,
        "train_cpu_time_s": train_meas.cpu_time_s,
        "train_wall_time_s": train_meas.wall_time_s,
        "train_co2_g": train_meas.co2_grams_total,
        "train_co2_cpu_g": train_meas.co2_grams_cpu,
        "inf_energy_j": inf_meas.cpu_energy_j * inf_scale,
        "inf_total_energy_j": inf_meas.total_energy_j * inf_scale,
        "inf_cpu_time_s": inf_meas.cpu_time_s * inf_scale,
        "inf_wall_time_s": inf_meas.wall_time_s * inf_scale,
        "inf_co2_g": inf_meas.co2_grams_total * inf_scale,
        "inf_co2_cpu_g": inf_meas.co2_grams_cpu * inf_scale,
        **metrics,
    }


def _select_balanced_grid(configs: List[Dict[str, Any]], max_configs: int) -> List[Dict[str, Any]]:
    """Select a Latin-rotated factorial subset preserving ALL hidden sizes, activations, solvers, alphas, and max_iters."""
    if len(configs) <= max_configs:
        return configs

    buckets: Dict[Tuple, List[Dict[str, Any]]] = {}
    for c in configs:
        key = (c["hidden_layer_sizes"], c["activation"], c["solver"])
        buckets.setdefault(key, []).append(c)

    sorted_keys = sorted(buckets.keys())
    selected: List[Dict[str, Any]] = []
    seen_ids = set()
    round_idx = 0
    while len(selected) < max_configs:
        added_any = False
        for b_idx, key in enumerate(sorted_keys):
            items = buckets[key]
            # Spread picks across different alpha and max_iter combinations per bucket
            idx = (b_idx * 3 + round_idx) % len(items)
            cand = items[idx]
            if cand["config_id"] not in seen_ids:
                selected.append(cand)
                seen_ids.add(cand["config_id"])
                added_any = True
                if len(selected) >= max_configs:
                    break
        if not added_any:
            break
        round_idx += 1
    return selected


def run_experiment_1_grid_search(
    dataset_key: str,
    profiler: EnergyProfiler,
    n_repeats: int = 10,
    inference_repeats: int = INFERENCE_REPEATS,
    compute_cv: bool = True,
    max_configs: Optional[int] = None,
    mortgage_max_rows: Optional[int] = None,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """Experiment 1: Exhaustive Grid Search with Algorithm 1 Interleaving + Pareto/Knee analysis."""
    ds = load_dataset(dataset_key, mortgage_max_rows=mortgage_max_rows)
    grid = MORTGAGE_MLP_GRID if dataset_key == "mortgage" else UCI_MLP_GRID
    configs = expand_mlp_grid(grid)
    if max_configs is not None and len(configs) > max_configs:
        configs = _select_balanced_grid(configs, max_configs)

    # Compute 5-fold CV accuracy once per config (avoids redundant 5xN deterministic repeats)
    cv_scores: Dict[int, float] = {}
    if compute_cv:
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for c in configs:
                m_cv = MLPClassifier(
                    hidden_layer_sizes=c["hidden_layer_sizes"],
                    activation=c["activation"],
                    solver=c["solver"],
                    alpha=float(c["alpha"]),
                    max_iter=int(c["max_iter"]),
                    random_state=RANDOM_SEED,
                )
                cv_scores[c["config_id"]] = float(np.mean(cross_val_score(m_cv, ds.X_train, ds.y_train, cv=skf)))

    raw_rows: List[Dict[str, Any]] = []
    # Algorithm 1: Outer loop over repeats, inner loop over all configs (interleaving)
    with profiler.track_codecarbon_session(f"exp1_grid_{dataset_key}") as cc_session:
        for rep in range(n_repeats):
            for c in configs:
                row_meas = _fit_and_measure_once(
                    ds=ds,
                    cfg=c,
                    profiler=profiler,
                    inference_repeats=inference_repeats,
                    seed=RANDOM_SEED + rep,
                )
                raw_rows.append(
                    {
                        "dataset": dataset_key,
                        "repeat": rep,
                        "config_id": c["config_id"],
                        "hidden_neurons": int(c["hidden_layer_sizes"][0]),
                        "activation": c["activation"],
                        "solver": c["solver"],
                        "alpha": float(c["alpha"]),
                        "max_iter": int(c["max_iter"]),
                        "cv_accuracy": cv_scores.get(c["config_id"], np.nan),
                        **row_meas,
                    }
                )

    raw_df = pd.DataFrame(raw_rows)

    # Aggregate with non-parametric Median and IQR (Paper Section IV-A & IV-B)
    agg_rows: List[Dict[str, Any]] = []
    for cid, grp in raw_df.groupby("config_id", sort=True):
        first = grp.iloc[0]
        agg_rows.append(
            {
                "dataset": dataset_key,
                "config_id": int(cid),
                "hidden_neurons": int(first["hidden_neurons"]),
                "activation": str(first["activation"]),
                "solver": str(first["solver"]),
                "alpha": float(first["alpha"]),
                "max_iter": int(first["max_iter"]),
                "cv_accuracy": float(first["cv_accuracy"]),
                "test_accuracy": float(grp["test_accuracy"].median()),
                "test_accuracy_iqr": _iqr(grp["test_accuracy"].to_numpy()),
                "precision": float(grp["precision"].median()),
                "recall": float(grp["recall"].median()),
                "f1_score": float(grp["f1_score"].median()),
                "train_energy_j": float(grp["train_energy_j"].median()),
                "train_energy_iqr": _iqr(grp["train_energy_j"].to_numpy()),
                "train_total_energy_j": float(grp["train_total_energy_j"].median()),
                "train_cpu_time_s": float(grp["train_cpu_time_s"].median()),
                "train_wall_time_s": float(grp["train_wall_time_s"].median()),
                "train_co2_g": float(grp["train_co2_g"].median()),
                "inf_energy_j": float(grp["inf_energy_j"].median()),
                "inf_energy_iqr": _iqr(grp["inf_energy_j"].to_numpy()),
                "inf_total_energy_j": float(grp["inf_total_energy_j"].median()),
                "inf_cpu_time_s": float(grp["inf_cpu_time_s"].median()),
                "inf_wall_time_s": float(grp["inf_wall_time_s"].median()),
                "inf_co2_g": float(grp["inf_co2_g"].median()),
            }
        )
    agg_df = pd.DataFrame(agg_rows)
    agg_df = add_mlp_complexity_columns(ds, agg_df)

    pareto_train = extract_pareto_front(agg_df, acc_col="test_accuracy", energy_col="train_energy_j")
    pareto_inf = extract_pareto_front(agg_df, acc_col="test_accuracy", energy_col="inf_energy_j")
    agg_df, pareto_train, pareto_inf = compute_pareto_cv_if_missing(ds, agg_df, pareto_train, pareto_inf)
    knee_train = find_knee_point(pareto_train, acc_col="test_accuracy", energy_col="train_energy_j", co2_col="train_co2_g")
    knee_inf = find_knee_point(pareto_inf, acc_col="test_accuracy", energy_col="inf_energy_j", co2_col="inf_co2_g")

    # CPU Time vs CPU Energy short-vs-long run check (Paper Figs 2 & 3)
    median_cpu_t = float(agg_df["train_cpu_time_s"].median())
    short_df = agg_df[agg_df["train_cpu_time_s"] <= median_cpu_t]
    long_df = agg_df[agg_df["train_cpu_time_s"] > median_cpu_t]
    short_slope = float(np.polyfit(short_df["train_cpu_time_s"], short_df["train_energy_j"], 1)[0]) if len(short_df) >= 2 else 0.0
    long_slope = float(np.polyfit(long_df["train_cpu_time_s"], long_df["train_energy_j"], 1)[0]) if len(long_df) >= 2 else 0.0

    if save_outputs:
        raw_df.to_csv(RESULTS_DIR / f"exp1_raw_{dataset_key}.csv", index=False)
        agg_df.to_csv(RESULTS_DIR / f"exp1_agg_{dataset_key}.csv", index=False)
        pareto_train.to_csv(RESULTS_DIR / f"exp1_pareto_train_{dataset_key}.csv", index=False)
        pareto_inf.to_csv(RESULTS_DIR / f"exp1_pareto_inf_{dataset_key}.csv", index=False)

    return {
        "dataset": ds,
        "raw_df": raw_df,
        "agg_df": agg_df,
        "pareto_train": pareto_train,
        "pareto_inf": pareto_inf,
        "knee_train": knee_train,
        "knee_inf": knee_inf,
        "short_run_slope_j_per_s": short_slope,
        "long_run_slope_j_per_s": long_slope,
        "codecarbon_session": cc_session,
        "n_configs": len(configs),
        "n_repeats": n_repeats,
    }


def add_mlp_complexity_columns(ds: PreparedDataset, df: pd.DataFrame) -> pd.DataFrame:
    """Add exact MLP parameter count and per-sample inference FLOPs (Paper Section VII proxy)."""
    if df.empty or "hidden_neurons" not in df.columns:
        return df
    out = df.copy()
    d_in = int(ds.encoded_features)
    c_out = int(ds.n_classes) if ds.n_classes > 2 else 1
    h = out["hidden_neurons"].astype(int)
    out["param_count"] = (d_in * h + h) + (h * c_out + c_out)
    out["inf_flops_per_sample"] = 2 * (d_in * h + h * c_out)
    return out


def compute_pareto_cv_if_missing(
    ds: PreparedDataset,
    agg_df: pd.DataFrame,
    pareto_train: pd.DataFrame,
    pareto_inf: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ensure 5-fold CV accuracy is populated on Pareto configs to verify generalization (Section IV)."""
    pareto_cids = set(pareto_train["config_id"].tolist()) | set(pareto_inf["config_id"].tolist())
    missing_df = agg_df[agg_df["config_id"].isin(pareto_cids) & agg_df["cv_accuracy"].isna()]
    if not missing_df.empty:
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        cv_map: Dict[int, float] = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for _, r in missing_df.iterrows():
                m_cv = MLPClassifier(
                    hidden_layer_sizes=(int(r["hidden_neurons"]),),
                    activation=str(r["activation"]),
                    solver=str(r["solver"]),
                    alpha=float(r["alpha"]),
                    max_iter=int(r["max_iter"]),
                    random_state=RANDOM_SEED,
                )
                cv_map[int(r["config_id"])] = float(np.mean(cross_val_score(m_cv, ds.X_train, ds.y_train, cv=skf)))
        agg_df = agg_df.copy()
        for cid, val in cv_map.items():
            agg_df.loc[agg_df["config_id"] == cid, "cv_accuracy"] = val
        pareto_train = extract_pareto_front(agg_df, acc_col="test_accuracy", energy_col="train_energy_j")
        pareto_inf = extract_pareto_front(agg_df, acc_col="test_accuracy", energy_col="inf_energy_j")
    return agg_df, pareto_train, pareto_inf


def run_normality_audit_30_repeats(
    dataset_key: str,
    profiler: EnergyProfiler,
    n_repeats: int = 30,
    inference_repeats: int = INFERENCE_REPEATS,
) -> Dict[str, Any]:
    """Run exact 30 interleaved repeats on representative configs to verify Fig 1 non-normality."""
    ds = load_dataset(dataset_key)
    rep_configs = [
        {"config_id": 0, "hidden_layer_sizes": (10,), "activation": "relu", "solver": "adam", "alpha": 0.0001, "max_iter": 200},
        {"config_id": 1, "hidden_layer_sizes": (100,), "activation": "tanh", "solver": "adam", "alpha": 0.0005, "max_iter": 400},
        {"config_id": 2, "hidden_layer_sizes": (100,), "activation": "relu", "solver": "lbfgs", "alpha": 0.0001, "max_iter": 200},
        {"config_id": 3, "hidden_layer_sizes": (50,), "activation": "logistic", "solver": "sgd", "alpha": 0.001, "max_iter": 200},
    ]
    rows = []
    for rep in range(n_repeats):
        for c in rep_configs:
            meas = _fit_and_measure_once(ds, c, profiler, inference_repeats, RANDOM_SEED + rep)
            rows.append(
                {
                    "dataset": dataset_key,
                    "repeat": rep,
                    "config_id": c["config_id"],
                    "label": f"h={c['hidden_layer_sizes'][0]}, {c['activation']}, {c['solver']}",
                    **meas,
                }
            )
    df = pd.DataFrame(rows)
    shapiro_results = []
    for cid, grp in df.groupby("config_id"):
        vals = grp["train_energy_j"].to_numpy()
        stat, p_val = stats.shapiro(vals)
        shapiro_results.append(
            {
                "config_id": int(cid),
                "label": str(grp["label"].iloc[0]),
                "median_j": float(np.median(vals)),
                "iqr_j": _iqr(vals),
                "skewness": float(stats.skew(vals)),
                "shapiro_p": float(p_val),
                "is_non_normal": bool(p_val < 0.05),
            }
        )
    audit_df = pd.DataFrame(shapiro_results)
    df.to_csv(RESULTS_DIR / f"normality_30rep_{dataset_key}.csv", index=False)
    audit_df.to_csv(RESULTS_DIR / f"normality_summary_{dataset_key}.csv", index=False)
    return {"raw_30rep": df, "summary": audit_df}


def run_experiment_2_parameter_analysis(
    agg_df: pd.DataFrame,
    pareto_train: pd.DataFrame,
    pareto_inf: pd.DataFrame,
    dataset_key: str,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """Experiment 2: Hyperparameter driver analysis & Training vs Inference comparison (RQ2, RQ3, RQ4)."""
    num_params = ["hidden_neurons", "alpha", "max_iter"]
    targets = ["test_accuracy", "train_energy_j", "inf_energy_j"]

    corr_rows = []
    for p in num_params:
        for t in targets:
            rho, pval = stats.spearmanr(agg_df[p], agg_df[t])
            corr_rows.append(
                {
                    "parameter": p,
                    "target": t,
                    "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
                    "abs_rho": float(abs(rho)) if not np.isnan(rho) else 0.0,
                    "p_value": float(pval) if not np.isnan(pval) else 1.0,
                }
            )
    spearman_df = pd.DataFrame(corr_rows)

    # Categorical drivers: Kruskal-Wallis across solver and activation
    cat_rows = []
    for cat_col in ["solver", "activation"]:
        for t in targets:
            groups = [grp[t].to_numpy() for _, grp in agg_df.groupby(cat_col)]
            if len(groups) >= 2:
                h_stat, pval = stats.kruskal(*groups)
            else:
                h_stat, pval = 0.0, 1.0
            med_map = agg_df.groupby(cat_col)[t].median().to_dict()
            cat_rows.append(
                {
                    "parameter": cat_col,
                    "target": t,
                    "kruskal_h": float(h_stat),
                    "p_value": float(pval),
                    "group_medians": str({k: round(float(v), 5) for k, v in med_map.items()}),
                }
            )
    cat_df = pd.DataFrame(cat_rows)

    # RQ4: Training Energy vs Inference Energy correlation & quadrant check
    rq4_rho, rq4_p = stats.spearmanr(agg_df["train_energy_j"], agg_df["inf_energy_j"])
    med_tr = float(agg_df["train_energy_j"].median())
    med_in = float(agg_df["inf_energy_j"].median())
    low_tr_low_in = int(((agg_df["train_energy_j"] <= med_tr) & (agg_df["inf_energy_j"] <= med_in)).sum())
    low_tr_high_in = int(((agg_df["train_energy_j"] <= med_tr) & (agg_df["inf_energy_j"] > med_in)).sum())
    high_tr_low_in = int(((agg_df["train_energy_j"] > med_tr) & (agg_df["inf_energy_j"] <= med_in)).sum())
    high_tr_high_in = int(((agg_df["train_energy_j"] > med_tr) & (agg_df["inf_energy_j"] > med_in)).sum())

    # Compare highest accuracy subset inside high-train-low-inf vs low-train-low-inf
    best_acc_high_tr_low_in = float(
        agg_df.loc[(agg_df["train_energy_j"] > med_tr) & (agg_df["inf_energy_j"] <= med_in), "test_accuracy"].max()
    ) if high_tr_low_in > 0 else np.nan
    best_acc_low_tr_low_in = float(
        agg_df.loc[(agg_df["train_energy_j"] <= med_tr) & (agg_df["inf_energy_j"] <= med_in), "test_accuracy"].max()
    ) if low_tr_low_in > 0 else np.nan

    # Section V counter-intuitive check: compact MLP + longer training vs large MLP + shorter training
    compact_long = agg_df[(agg_df["hidden_neurons"] <= 20) & (agg_df["max_iter"] >= 400)]
    large_short = agg_df[(agg_df["hidden_neurons"] >= 100) & (agg_df["max_iter"] <= 200)]
    flops_rho = np.nan
    if "inf_flops_per_sample" in agg_df.columns:
        flops_rho, _ = stats.spearmanr(agg_df["inf_flops_per_sample"], agg_df["inf_energy_j"])

    rq4_summary = {
        "spearman_train_vs_inf": float(rq4_rho) if not np.isnan(rq4_rho) else 0.0,
        "p_value": float(rq4_p) if not np.isnan(rq4_p) else 1.0,
        "low_train_low_inf_count": low_tr_low_in,
        "low_train_high_inf_count": low_tr_high_in,
        "high_train_low_inf_count": high_tr_low_in,
        "high_train_high_inf_count": high_tr_high_in,
        "best_acc_high_train_low_inf": best_acc_high_tr_low_in,
        "best_acc_low_train_low_inf": best_acc_low_tr_low_in,
        "compact_long_best_acc": float(compact_long["test_accuracy"].max()) if not compact_long.empty else np.nan,
        "compact_long_median_inf_j": float(compact_long["inf_energy_j"].median()) if not compact_long.empty else np.nan,
        "large_short_best_acc": float(large_short["test_accuracy"].max()) if not large_short.empty else np.nan,
        "large_short_median_inf_j": float(large_short["inf_energy_j"].median()) if not large_short.empty else np.nan,
        "spearman_flops_vs_inf_energy": float(flops_rho) if not np.isnan(flops_rho) else 0.0,
    }

    if save_outputs:
        spearman_df.to_csv(RESULTS_DIR / f"exp2_spearman_{dataset_key}.csv", index=False)
        cat_df.to_csv(RESULTS_DIR / f"exp2_categorical_{dataset_key}.csv", index=False)
        pareto_train.sort_values(by=["train_energy_j", "test_accuracy"]).to_csv(
            RESULTS_DIR / f"exp2_pareto_sorted_train_{dataset_key}.csv", index=False
        )
        pareto_inf.sort_values(by=["inf_energy_j", "test_accuracy"]).to_csv(
            RESULTS_DIR / f"exp2_pareto_sorted_inf_{dataset_key}.csv", index=False
        )

    return {
        "spearman_df": spearman_df,
        "categorical_df": cat_df,
        "rq4_summary": rq4_summary,
    }


class _MLPEnergyProblem(FloatProblem):
    """jMetalPy Multi-Objective Problem: Minimize (1 - Test Accuracy) and Training Energy (J)."""

    def __init__(
        self,
        ds: PreparedDataset,
        profiler: EnergyProfiler,
        repeats_per_eval: int = 3,
        inference_repeats: int = 200,
        seed_offset: int = 0,
    ) -> None:
        super().__init__()
        self.ds = ds
        self.profiler = profiler
        self.repeats_per_eval = repeats_per_eval
        self.inference_repeats = inference_repeats
        self.seed_offset = seed_offset

        self.lower_bound = [
            float(NSGA2_CONFIG["hidden_min"]),
            0.0,
            0.0,
            float(NSGA2_CONFIG["alpha_min"]),
            float(NSGA2_CONFIG["max_iter_min"]),
        ]
        self.upper_bound = [
            float(NSGA2_CONFIG["hidden_max"]),
            2.999,
            2.999,
            float(NSGA2_CONFIG["alpha_max"]),
            float(NSGA2_CONFIG["max_iter_max"]),
        ]
        self.directions = [self.MINIMIZE, self.MINIMIZE]
        self.labels = ["Error (1 - Test Accuracy)", "Mean Training CPU Energy (J)"]
        self.eval_history: List[Dict[str, Any]] = []
        self._eval_counter = 0
        self._cache: Dict[Tuple, Tuple[float, float, float, float]] = {}

    def number_of_variables(self) -> int:
        return 5

    def number_of_objectives(self) -> int:
        return 2

    def number_of_constraints(self) -> int:
        return 0

    def name(self) -> str:
        return f"MLPAccuracyEnergy_{self.ds.key}"

    @staticmethod
    def decode(vars_list: List[float]) -> Dict[str, Any]:
        acts = NSGA2_CONFIG["activations"]
        sols = NSGA2_CONFIG["solvers"]
        h = int(np.clip(round(vars_list[0]), NSGA2_CONFIG["hidden_min"], NSGA2_CONFIG["hidden_max"]))
        act_idx = int(np.clip(int(np.floor(vars_list[1])), 0, len(acts) - 1))
        sol_idx = int(np.clip(int(np.floor(vars_list[2])), 0, len(sols) - 1))
        alpha = float(np.clip(vars_list[3], NSGA2_CONFIG["alpha_min"], NSGA2_CONFIG["alpha_max"]))
        max_it = int(np.clip(round(vars_list[4]), NSGA2_CONFIG["max_iter_min"], NSGA2_CONFIG["max_iter_max"]))
        return {
            "hidden_layer_sizes": (h,),
            "activation": acts[act_idx],
            "solver": sols[sol_idx],
            "alpha": alpha,
            "max_iter": max_it,
        }

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        cfg = self.decode(solution.variables)
        cache_key = (
            int(cfg["hidden_layer_sizes"][0]),
            cfg["activation"],
            cfg["solver"],
            round(float(cfg["alpha"]), 5),
            int(cfg["max_iter"]),
        )
        if cache_key in self._cache:
            mean_acc, mean_train_e, mean_inf_e, mean_co2 = self._cache[cache_key]
        else:
            train_energies = []
            inf_energies = []
            accs = []
            co2_vals = []

            for r in range(self.repeats_per_eval):
                meas = _fit_and_measure_once(
                    ds=self.ds,
                    cfg=cfg,
                    profiler=self.profiler,
                    inference_repeats=self.inference_repeats,
                    seed=RANDOM_SEED + self.seed_offset + r,
                )
                train_energies.append(meas["train_energy_j"])
                inf_energies.append(meas["inf_energy_j"])
                accs.append(meas["test_accuracy"])
                co2_vals.append(meas["train_co2_g"])

            mean_acc = float(np.mean(accs))
            mean_train_e = float(np.mean(train_energies))
            mean_inf_e = float(np.mean(inf_energies))
            mean_co2 = float(np.mean(co2_vals))
            self._cache[cache_key] = (mean_acc, mean_train_e, mean_inf_e, mean_co2)

        solution.objectives[0] = 1.0 - mean_acc
        solution.objectives[1] = mean_train_e

        self._eval_counter += 1
        if self._eval_counter % 16 == 0:
            print(f"    [NSGA-II {self.ds.key}] eval {self._eval_counter}: acc={mean_acc:.3f}, E={mean_train_e:.3f}J", flush=True)
        self.eval_history.append(
            {
                "eval_idx": self._eval_counter,
                "hidden_neurons": int(cfg["hidden_layer_sizes"][0]),
                "activation": cfg["activation"],
                "solver": cfg["solver"],
                "alpha": cfg["alpha"],
                "max_iter": cfg["max_iter"],
                "test_accuracy": mean_acc,
                "train_energy_j": mean_train_e,
                "inf_energy_j": mean_inf_e,
                "train_co2_g": mean_co2,
            }
        )
        return solution


def run_experiment_3_nsga2(
    dataset_key: str,
    profiler: EnergyProfiler,
    n_runs: int = 5,
    pop_size: int = NSGA2_CONFIG["population_size"],
    max_evaluations: int = NSGA2_CONFIG["max_evaluations"],
    repeats_per_eval: int = 3,
    grid_agg_df: Optional[pd.DataFrame] = None,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """Experiment 3: NSGA-II via jMetalPy to approximate the Accuracy-Energy Pareto front."""
    ds = load_dataset(dataset_key)
    all_runs_hist: List[pd.DataFrame] = []
    all_runs_fronts: List[pd.DataFrame] = []

    with profiler.track_codecarbon_session(f"exp3_nsga2_{dataset_key}") as cc_session:
        for run_idx in range(n_runs):
            rng = np.random.default_rng(RANDOM_SEED + 100 * run_idx)
            problem = _MLPEnergyProblem(
                ds=ds,
                profiler=profiler,
                repeats_per_eval=repeats_per_eval,
                inference_repeats=1,
                seed_offset=run_idx * 10,
            )
            algorithm = NSGAII(
                problem=problem,
                population_size=pop_size,
                offspring_population_size=pop_size,
                mutation=PolynomialMutation(
                    probability=NSGA2_CONFIG["mutation_probability"],
                    distribution_index=NSGA2_CONFIG["mutation_distribution_index"],
                ),
                crossover=SBXCrossover(
                    probability=NSGA2_CONFIG["crossover_probability"],
                    distribution_index=NSGA2_CONFIG["crossover_distribution_index"],
                ),
                termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
                rng=rng,
            )
            algorithm.run()

            hist_df = pd.DataFrame(problem.eval_history)
            hist_df["run_id"] = run_idx
            all_runs_hist.append(hist_df)

            pf_run = extract_pareto_front(hist_df, acc_col="test_accuracy", energy_col="train_energy_j")
            pf_run["run_id"] = run_idx
            all_runs_fronts.append(pf_run)

    combined_hist = pd.concat(all_runs_hist, ignore_index=True)
    attainment_df = compute_attainment_curves(
        all_runs_fronts, acc_col="test_accuracy", energy_col="train_energy_j", n_grid=100
    )

    # Hypervolume comparison against Grid Search on a shared reference energy
    ref_energy = float(combined_hist["train_energy_j"].max() * 1.1)
    if grid_agg_df is not None and not grid_agg_df.empty:
        ref_energy = max(ref_energy, float(grid_agg_df["train_energy_j"].max() * 1.1))
        grid_pf = extract_pareto_front(grid_agg_df, acc_col="test_accuracy", energy_col="train_energy_j")
        hv_grid = compute_hypervolume_2d(grid_pf, acc_col="test_accuracy", energy_col="train_energy_j", ref_energy=ref_energy)
    else:
        grid_pf = pd.DataFrame()
        hv_grid = 0.0

    hv_runs = [
        compute_hypervolume_2d(pf, acc_col="test_accuracy", energy_col="train_energy_j", ref_energy=ref_energy)
        for pf in all_runs_fronts
    ]
    hv_median = float(np.median(hv_runs)) if hv_runs else 0.0
    hv_best = float(np.max(hv_runs)) if hv_runs else 0.0

    if save_outputs:
        combined_hist.to_csv(RESULTS_DIR / f"exp3_nsga2_history_{dataset_key}.csv", index=False)
        attainment_df.to_csv(RESULTS_DIR / f"exp3_attainment_{dataset_key}.csv", index=False)

    return {
        "history_df": combined_hist,
        "run_fronts": all_runs_fronts,
        "attainment_df": attainment_df,
        "grid_pareto": grid_pf,
        "hv_grid": hv_grid,
        "hv_nsga2_median": hv_median,
        "hv_nsga2_best": hv_best,
        "hv_ratio_median": (hv_median / hv_grid) if hv_grid > 0 else 0.0,
        "hv_ratio_best": (hv_best / hv_grid) if hv_grid > 0 else 0.0,
        "codecarbon_session": cc_session,
        "max_evaluations": max_evaluations,
        "n_runs": n_runs,
    }
