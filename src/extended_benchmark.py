"""Connected extensions: Proposal Topic 1 classifiers, Paper Future Work, Carbon Payback, and Pareto Hot-Swap Surface."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import MODELS_DIR, RANDOM_SEED, RESULTS_DIR
from src.data_loader import PreparedDataset, load_dataset
from src.energy_profiler import EnergyProfiler
from src.pareto import extract_pareto_front, find_knee_point


def _build_model_catalog(include_sweeps: bool = True) -> List[Dict[str, Any]]:
    """Build cross-paradigm classifier configurations (Proposal Topic 1 + Paper Future Work)."""
    catalog: List[Dict[str, Any]] = []

    if include_sweeps:
        for c_val in [0.01, 0.1, 1.0, 10.0, 100.0]:
            catalog.append({
                "family": "LogisticRegression",
                "variant": f"C={c_val}",
                "builder": lambda seed, c=c_val: LogisticRegression(C=c, max_iter=1000, random_state=seed),
            })
        for depth in [2, 4, 6, 10, None]:
            catalog.append({
                "family": "DecisionTree",
                "variant": f"depth={depth}",
                "builder": lambda seed, d=depth: DecisionTreeClassifier(max_depth=d, random_state=seed),
            })
        for n_est in [10, 25, 50, 100, 200]:
            for depth in [5, None]:
                catalog.append({
                    "family": "RandomForest",
                    "variant": f"n={n_est},d={depth}",
                    "builder": lambda seed, n=n_est, d=depth: RandomForestClassifier(
                        n_estimators=n, max_depth=d, random_state=seed, n_jobs=1
                    ),
                })
        for kernel in ["linear", "rbf"]:
            for c_val in [0.1, 1.0, 10.0]:
                catalog.append({
                    "family": "SVM",
                    "variant": f"{kernel},C={c_val}",
                    "builder": lambda seed, k=kernel, c=c_val: SVC(kernel=k, C=c, random_state=seed),
                })
        for k_nb in [1, 3, 5, 11, 21]:
            catalog.append({
                "family": "KNN",
                "variant": f"k={k_nb}",
                "builder": lambda seed, k=k_nb: KNeighborsClassifier(n_neighbors=k),
            })
        for h in [5, 20, 50, 100]:
            catalog.append({
                "family": "MLP",
                "variant": f"h={h},adam,relu",
                "builder": lambda seed, hid=h: MLPClassifier(
                    hidden_layer_sizes=(hid,), activation="relu", solver="adam", max_iter=150, random_state=seed
                ),
            })
    else:
        catalog = [
            {"family": "LogisticRegression", "variant": "default", "builder": lambda seed: LogisticRegression(max_iter=1000, random_state=seed)},
            {"family": "DecisionTree", "variant": "default", "builder": lambda seed: DecisionTreeClassifier(random_state=seed)},
            {"family": "RandomForest", "variant": "n=100", "builder": lambda seed: RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)},
            {"family": "SVM", "variant": "rbf,C=1.0", "builder": lambda seed: SVC(kernel="rbf", C=1.0, random_state=seed)},
            {"family": "KNN", "variant": "k=5", "builder": lambda seed: KNeighborsClassifier(n_neighbors=5)},
            {"family": "MLP", "variant": "h=100,adam", "builder": lambda seed: MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=seed)},
        ]
    return catalog


def run_cross_model_benchmark(
    dataset_key: str,
    profiler: EnergyProfiler,
    n_repeats: int = 5,
    inference_repeats: int = 200,
    include_sweeps: bool = True,
    mortgage_max_rows: Optional[int] = 50000,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """Benchmark Logistic Regression, Decision Tree, Random Forest, SVM, KNN, and MLP."""
    ds = load_dataset(dataset_key, mortgage_max_rows=mortgage_max_rows)
    catalog = _build_model_catalog(include_sweeps=include_sweeps)

    # For medium/large datasets (>3k rows), avoid slow redundant RBF SVM & 200x kernel inference overhead
    if len(ds.X_train) > 3000:
        catalog = [
            c for c in catalog
            if not (c["family"] == "SVM" and "rbf" in c["variant"] and c["variant"] != "rbf,C=1.0")
        ]
        inference_repeats = min(inference_repeats, 30)

    raw_rows: List[Dict[str, Any]] = []
    with profiler.track_codecarbon_session(f"cross_models_{dataset_key}") as cc_session:
        for rep in range(n_repeats):
            for idx, entry in enumerate(catalog):
                model = entry["builder"](RANDOM_SEED + rep)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    _, tr_m = profiler.measure_callable(lambda: model.fit(ds.X_train, ds.y_train))

                    def _inf() -> np.ndarray:
                        preds = None
                        for _ in range(inference_repeats):
                            preds = model.predict(ds.X_test)
                        return preds

                    y_pred, inf_m = profiler.measure_callable(_inf)

                avg = "binary" if ds.n_classes == 2 else "weighted"
                inf_scale = 1000.0 / max(1, inference_repeats)
                raw_rows.append(
                    {
                        "dataset": dataset_key,
                        "model_id": idx,
                        "family": entry["family"],
                        "variant": entry["variant"],
                        "model_label": f"{entry['family']} ({entry['variant']})",
                        "repeat": rep,
                        "test_accuracy": float(accuracy_score(ds.y_test, y_pred)),
                        "precision": float(precision_score(ds.y_test, y_pred, average=avg, zero_division=0)),
                        "recall": float(recall_score(ds.y_test, y_pred, average=avg, zero_division=0)),
                        "f1_score": float(f1_score(ds.y_test, y_pred, average=avg, zero_division=0)),
                        "train_time_s": tr_m.wall_time_s,
                        "train_cpu_time_s": tr_m.cpu_time_s,
                        "train_energy_j": tr_m.cpu_energy_j,
                        "train_energy_kwh": tr_m.total_energy_kwh,
                        "train_co2_g": tr_m.co2_grams_total,
                        "inf_time_1k_s": inf_m.wall_time_s * inf_scale,
                        "inf_energy_j": inf_m.cpu_energy_j * inf_scale,
                        "inf_energy_kwh": inf_m.total_energy_kwh * inf_scale,
                        "inf_co2_g": inf_m.co2_grams_total * inf_scale,
                    }
                )

    raw_df = pd.DataFrame(raw_rows)
    agg_df = (
        raw_df.groupby(["dataset", "model_id", "family", "variant", "model_label"], as_index=False)
        .agg(
            test_accuracy=("test_accuracy", "median"),
            precision=("precision", "median"),
            recall=("recall", "median"),
            f1_score=("f1_score", "median"),
            train_time_s=("train_time_s", "median"),
            train_cpu_time_s=("train_cpu_time_s", "median"),
            train_energy_j=("train_energy_j", "median"),
            train_energy_kwh=("train_energy_kwh", "median"),
            train_co2_g=("train_co2_g", "median"),
            inf_time_1k_s=("inf_time_1k_s", "median"),
            inf_energy_j=("inf_energy_j", "median"),
            inf_energy_kwh=("inf_energy_kwh", "median"),
            inf_co2_g=("inf_co2_g", "median"),
        )
    )

    pareto_train = extract_pareto_front(agg_df, acc_col="test_accuracy", energy_col="train_energy_j")
    pareto_inf = extract_pareto_front(agg_df, acc_col="test_accuracy", energy_col="inf_energy_j")
    knee_train = find_knee_point(pareto_train, acc_col="test_accuracy", energy_col="train_energy_j", co2_col="train_co2_g")
    knee_inf = find_knee_point(pareto_inf, acc_col="test_accuracy", energy_col="inf_energy_j", co2_col="inf_co2_g")

    if save_outputs:
        agg_df.to_csv(RESULTS_DIR / f"cross_model_{dataset_key}.csv", index=False)
        pareto_train.to_csv(RESULTS_DIR / f"cross_model_pareto_train_{dataset_key}.csv", index=False)
        pareto_inf.to_csv(RESULTS_DIR / f"cross_model_pareto_inf_{dataset_key}.csv", index=False)

    return {
        "dataset": ds,
        "raw_df": raw_df,
        "agg_df": agg_df,
        "pareto_train": pareto_train,
        "pareto_inf": pareto_inf,
        "knee_train": knee_train,
        "knee_inf": knee_inf,
        "codecarbon_session": cc_session,
    }


def compute_carbon_payback(
    grid_search_energy_j: float,
    nsga2_search_energy_j: float,
    baseline_inf_energy_1k_j: float,
    green_inf_energy_1k_j: float,
    n_test_samples: int,
    carbon_intensity_kg_per_kwh: float = 0.4406,
    daily_inferences: int = 100000,
) -> Dict[str, float]:
    """Second-order Carbon Payback analysis: when does tuning pay back its own search energy?"""
    # Convert 1k test-batch energy into per-single-sample inference energy (J)
    base_per_sample_j = baseline_inf_energy_1k_j / max(1, 1000 * n_test_samples)
    green_per_sample_j = green_inf_energy_1k_j / max(1, 1000 * n_test_samples)
    saved_per_sample_j = max(0.0, base_per_sample_j - green_per_sample_j)

    if saved_per_sample_j > 0:
        breakeven_grid_inf = grid_search_energy_j / saved_per_sample_j
        breakeven_nsga2_inf = nsga2_search_energy_j / saved_per_sample_j
        days_grid = breakeven_grid_inf / max(1, daily_inferences)
        days_nsga2 = breakeven_nsga2_inf / max(1, daily_inferences)
    else:
        breakeven_grid_inf = float("inf")
        breakeven_nsga2_inf = float("inf")
        days_grid = float("inf")
        days_nsga2 = float("inf")

    co2_saved_per_1m_g = (saved_per_sample_j * 1_000_000.0 / 3_600_000.0) * carbon_intensity_kg_per_kwh * 1000.0
    return {
        "grid_search_energy_j": float(grid_search_energy_j),
        "nsga2_search_energy_j": float(nsga2_search_energy_j),
        "saved_per_sample_uj": float(saved_per_sample_j * 1e6),
        "co2_saved_per_1m_inferences_g": float(co2_saved_per_1m_g),
        "breakeven_inferences_grid": float(breakeven_grid_inf),
        "breakeven_inferences_nsga2": float(breakeven_nsga2_inf),
        "breakeven_days_grid": float(days_grid),
        "breakeven_days_nsga2": float(days_nsga2),
    }


def export_pareto_hotswap_surface(
    dataset_key: str,
    cross_df: pd.DataFrame,
    profiler: EnergyProfiler,
    mortgage_max_rows: Optional[int] = 50000,
) -> Dict[str, Any]:
    """Build and persist a deployable Pareto Program Surface bundle for runtime hot-swapping."""
    ds = load_dataset(dataset_key, mortgage_max_rows=mortgage_max_rows)
    pf = extract_pareto_front(cross_df, acc_col="test_accuracy", energy_col="inf_energy_j")
    if pf.empty:
        return {}

    catalog_map = {f"{e['family']} ({e['variant']})": e["builder"] for e in _build_model_catalog(include_sweeps=True)}

    surface_entries: List[Dict[str, Any]] = []
    trained_models: Dict[str, Any] = {}

    for rank_idx, (_, row) in enumerate(pf.iterrows()):
        label = str(row["model_label"])
        builder = catalog_map.get(label)
        if builder is None:
            continue
        model = builder(RANDOM_SEED)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(ds.X_train, ds.y_train)

        # Tier name for intuitive UI/runtime switching
        if rank_idx == 0:
            tier = "Ultra-Green"
        elif rank_idx == len(pf) - 1:
            tier = "Max-Accuracy"
        else:
            tier = f"Balanced-{rank_idx}"

        model_key = f"tier_{rank_idx}"
        trained_models[model_key] = model
        surface_entries.append(
            {
                "model_key": model_key,
                "tier": tier,
                "model_label": label,
                "family": str(row["family"]),
                "test_accuracy": float(row["test_accuracy"]),
                "f1_score": float(row["f1_score"]),
                "inf_energy_1k_j": float(row["inf_energy_j"]),
                "inf_co2_1k_g": float(row["inf_co2_g"]),
                "train_energy_j": float(row["train_energy_j"]),
            }
        )

    bundle_path = MODELS_DIR / f"{dataset_key}_pareto_surface.joblib"
    manifest_path = MODELS_DIR / f"{dataset_key}_pareto_manifest.json"
    joblib.dump(
        {
            "models": trained_models,
            "entries": surface_entries,
            "feature_names": ds.feature_names,
            "mortgage_max_rows": mortgage_max_rows,
        },
        bundle_path,
    )
    manifest_path.write_text(json.dumps(surface_entries, indent=2), encoding="utf-8")

    return {"bundle_path": str(bundle_path), "manifest": surface_entries}


def select_hotswap_model(
    entries: List[Dict[str, Any]],
    battery_pct: float = 100.0,
    min_accuracy: float = 0.0,
) -> Dict[str, Any]:
    """Runtime Adaptive Policy Router: pick highest-accuracy Pareto model fitting battery/SLA budget."""
    if not entries:
        raise ValueError("Empty Pareto surface")

    ordered = sorted(entries, key=lambda e: e["inf_energy_1k_j"])
    min_e = ordered[0]["inf_energy_1k_j"]
    max_e = ordered[-1]["inf_energy_1k_j"]

    # Map battery percentage (10% -> tightest energy budget, 100% -> full energy allowed)
    budget_frac = np.clip((battery_pct - 10.0) / 90.0, 0.0, 1.0)
    max_allowed_energy = min_e + budget_frac * (max_e - min_e)

    feasible = [
        e for e in ordered
        if e["inf_energy_1k_j"] <= max_allowed_energy + 1e-12 and e["test_accuracy"] >= min_accuracy
    ]
    if feasible:
        return max(feasible, key=lambda e: e["test_accuracy"])

    # If SLA min_accuracy cannot be satisfied under battery cap, fallback to accuracy-first or greenest
    acc_ok = [e for e in ordered if e["test_accuracy"] >= min_accuracy]
    if acc_ok:
        return min(acc_ok, key=lambda e: e["inf_energy_1k_j"])
    return ordered[0]
