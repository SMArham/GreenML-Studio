"""FastAPI REST Backend separating Frontend UI from Data & ML/Energy Engine."""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from src.config import FIGURES_DIR, MODELS_DIR, RANDOM_SEED, RESULTS_DIR, ROOT_DIR
from src.data_loader import load_dataset
from src.energy_profiler import EnergyProfiler
from src.extended_benchmark import select_hotswap_model
from src.pareto import extract_pareto_front, find_knee_point

FRONTEND_DIR = ROOT_DIR / "frontend"
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="GreenML Studio API",
    description="Empirical Machine Learning Accuracy and Energy Trade-Off Engine",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FIGURES_DIR.exists():
    app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

_PROFILER: Optional[EnergyProfiler] = None
_BUNDLE_CACHE: Dict[str, Dict[str, Any]] = {}


def _get_profiler() -> EnergyProfiler:
    global _PROFILER
    if _PROFILER is None:
        _PROFILER = EnergyProfiler(country_iso_code="PAK")
    return _PROFILER


def _clean_json_val(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _clean_json_val(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_clean_json_val(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _df_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    return _clean_json_val(df.to_dict(orient="records"))


def _read_summary() -> Dict[str, Any]:
    path = RESULTS_DIR / "master_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_safe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


class HotSwapPredictRequest(BaseModel):
    dataset: str = Field(..., description="Dataset key with exported Pareto bundle")
    battery_pct: float = Field(50.0, ge=10.0, le=100.0)
    min_accuracy: float = Field(0.0, ge=0.0, le=1.0)
    sample_idx: int = Field(0, ge=0)
    batch_repeats: int = Field(100, ge=10, le=500)


class LiveExperimentRequest(BaseModel):
    dataset: str = Field("diabetes", description="iris, glass, ionosphere, diabetes, telco_churn, titanic")
    mlp_hidden: int = Field(50, ge=2, le=500)
    mlp_solver: str = Field("adam", description="sgd, adam, lbfgs")
    mlp_activation: str = Field("relu", description="logistic, tanh, relu")


@app.get("/", include_in_schema=False)
def serve_frontend() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found")
    return FileResponse(index_path, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/favicon.ico", include_in_schema=False)
def serve_favicon() -> FileResponse:
    fav_ico = FRONTEND_DIR / "favicon.ico"
    if fav_ico.exists():
        return FileResponse(fav_ico, media_type="image/x-icon")
    fav_svg = FRONTEND_DIR / "favicon.svg"
    if fav_svg.exists():
        return FileResponse(fav_svg, media_type="image/svg+xml")
    raise HTTPException(status_code=404, detail="Favicon not found")


@app.get("/api/health")
def health() -> Dict[str, Any]:
    summary = _read_summary()
    return _clean_json_val({
        "status": "ok",
        "has_summary": bool(summary),
        "paper_datasets": list(summary.get("paper_datasets", {}).keys()),
        "nsga2_datasets": list(summary.get("nsga2_experiments", {}).keys()),
        "cross_model_datasets": list(summary.get("cross_model_benchmarks", {}).keys()),
        "hotswap_datasets": [k for k, v in summary.get("hotswap_bundles", {}).items() if v],
        "hardware": summary.get("hardware", {}),
    })


@app.get("/api/overview")
def get_overview() -> Dict[str, Any]:
    summary = _read_summary()
    if not summary:
        raise HTTPException(status_code=404, detail="master_summary.json not generated yet")
    return _clean_json_val(summary)


FIGURE_METADATA: Dict[str, Dict[str, str]] = {
    "01_master_research_synthesis.png": {
        "title": "Master Research Synthesis: The Full Empirical Story in One View",
        "short_title": "01 · Master All-in-One Synthesis",
        "category": "All-in-One Dashboard",
        "badge": "Core Master Plate",
        "description": "Multi-panel executive dashboard consolidating Pareto frontier knee points, hyperparameter attribution, NSGA-II hypervolume attainment, carbon payback horizons, and cross-model classifier benchmarks in a single publication-grade visualization.",
        "takeaway": "Empirically proves that optimal knee points save up to 85.7% energy for <1.5% accuracy trade-off across all classification tasks.",
    },
    "02_pareto_frontiers_multi_dataset.png": {
        "title": "Multi-Dataset Pareto Frontiers & Knee Point Trajectories",
        "short_title": "02 · Multi-Dataset Pareto Frontiers",
        "category": "Experiment 1",
        "badge": "Pareto Frontiers",
        "description": "Comparative non-dominated Pareto frontiers across UCI datasets (Diabetes, Ionosphere, Glass, Iris, Mortgage) mapping test accuracy against CPU energy consumption in both training and inference lifecycle stages.",
        "takeaway": "Identifies non-dominated boundary curves and mathematical knee points, showing steep initial energy climb for diminishing accuracy returns.",
    },
    "03_hyperparameter_drivers_decoupling.png": {
        "title": "Hyperparameter Attribution: Spearman Rank Correlation",
        "short_title": "03 · Hyperparameter Attribution",
        "category": "Experiment 2",
        "badge": "Spearman Drivers",
        "description": "Empirical statistical attribution analyzing numerical architecture parameters (Spearman rank correlation) across training and inference lifecycle stages.",
        "takeaway": "Proves hidden neurons dominate inference energy (rho = 0.801) and max iterations drive training energy (rho = 0.586), while regularization (alpha) has virtually no inference impact.",
    },
    "04_nsga2_payback_synthesis.png": {
        "title": "NSGA-II Multi-Objective Evolutionary Search & Carbon Payback",
        "short_title": "04 · NSGA-II & Carbon Payback",
        "category": "Experiment 3",
        "badge": "NSGA-II & Payback",
        "description": "Empirical attainment surfaces comparing NSGA-II genetic algorithm against exhaustive grid search, paired with production carbon payback break-even trajectory curves.",
        "takeaway": "NSGA-II discovers 99.1% of the true grid hypervolume in only 48 evaluations (saving 83% search compute), achieving carbon break-even in 352 days vs. 5.6 years for grid search.",
    },
    "05_cross_model_benchmark_synthesis.png": {
        "title": "Cross-Paradigm Classifier Benchmark: Multi-Family Frontiers",
        "short_title": "05 · Cross-Model Benchmark",
        "category": "Topic 1 Extension",
        "badge": "Cross-Model",
        "description": "Cross-model evaluation across 6 classifier families (Decision Trees, Random Forests, Logistic Regression, Linear & RBF SVMs, and KNN) on multi-domain tabular datasets.",
        "takeaway": "Demonstrates that simpler models like Logistic Regression and KNN form green knee solutions, saving up to 96.4% energy compared to heavy ensembles like Random Forest.",
    },
}


@app.get("/api/figures")
def list_figures() -> Dict[str, Any]:
    if not FIGURES_DIR.exists():
        return {"figures": [], "items": []}
    files = sorted([f.name for f in FIGURES_DIR.glob("*.png")])
    items = []
    for f in files:
        meta = FIGURE_METADATA.get(f, {
            "title": f.replace("_", " ").replace(".png", "").title(),
            "short_title": f,
            "category": "Publication Figure",
            "badge": "Empirical Result",
            "description": "Empirical publication figure from the GreenML experimental benchmark.",
            "takeaway": "Visual evidence supporting multi-objective accuracy and energy trade-off optimization.",
        })
        items.append({
            "id": f,
            "url": f"/figures/{f}",
            **meta
        })
    return {"figures": files, "items": items}


@app.get("/api/exp1/{dataset}")
def get_exp1(dataset: str) -> Dict[str, Any]:
    agg_df = _read_csv_safe(RESULTS_DIR / f"exp1_agg_{dataset}.csv")
    if agg_df.empty:
        raise HTTPException(status_code=404, detail=f"Experiment 1 data missing for {dataset}")

    pf_train = extract_pareto_front(agg_df, acc_col="test_accuracy", energy_col="train_energy_j")
    pf_inf = extract_pareto_front(agg_df, acc_col="test_accuracy", energy_col="inf_energy_j")
    knee_train = find_knee_point(pf_train, acc_col="test_accuracy", energy_col="train_energy_j", co2_col="train_co2_g")
    knee_inf = find_knee_point(pf_inf, acc_col="test_accuracy", energy_col="inf_energy_j", co2_col="inf_co2_g")

    summary = _read_summary()
    meta = summary.get("paper_datasets", {}).get(dataset, {})

    norm_key = dataset if dataset in summary.get("normality_audit", {}) else "diabetes"
    raw_30 = _read_csv_safe(RESULTS_DIR / f"normality_30rep_{norm_key}.csv")
    sum_30 = _read_csv_safe(RESULTS_DIR / f"normality_summary_{norm_key}.csv")
    hist_values: List[float] = []
    norm_meta: Dict[str, Any] = {}
    if not raw_30.empty and not sum_30.empty:
        cid0 = int(sum_30["config_id"].iloc[0])
        hist_values = raw_30.loc[raw_30["config_id"] == cid0, "train_energy_j"].astype(float).tolist()
        norm_meta = sum_30.iloc[0].to_dict()

    return _clean_json_val({
        "dataset": dataset,
        "meta": meta,
        "configs": _df_records(agg_df),
        "pareto_train": _df_records(pf_train),
        "pareto_inf": _df_records(pf_inf),
        "knee_train": knee_train.__dict__ if knee_train else None,
        "knee_inf": knee_inf.__dict__ if knee_inf else None,
        "normality_audit": {
            "dataset": norm_key,
            "values": hist_values,
            "summary": norm_meta,
        },
    })


@app.get("/api/exp2/{dataset}")
def get_exp2(dataset: str) -> Dict[str, Any]:
    agg_df = _read_csv_safe(RESULTS_DIR / f"exp1_agg_{dataset}.csv")
    if agg_df.empty:
        raise HTTPException(status_code=404, detail=f"Experiment 2 data missing for {dataset}")

    pf_train = _read_csv_safe(RESULTS_DIR / f"exp2_pareto_sorted_train_{dataset}.csv")
    pf_inf = _read_csv_safe(RESULTS_DIR / f"exp2_pareto_sorted_inf_{dataset}.csv")
    sp_df = _read_csv_safe(RESULTS_DIR / f"exp2_spearman_{dataset}.csv")
    cat_df = _read_csv_safe(RESULTS_DIR / f"exp2_categorical_{dataset}.csv")
    summary = _read_summary()
    rq4 = summary.get("paper_datasets", {}).get(dataset, {}).get("rq4_summary", {})

    extra_cols = [c for c in ["param_count", "inf_flops_per_sample"] if c in agg_df.columns]
    return _clean_json_val({
        "dataset": dataset,
        "pareto_sorted_train": _df_records(pf_train),
        "pareto_sorted_inf": _df_records(pf_inf),
        "spearman": _df_records(sp_df),
        "categorical": _df_records(cat_df),
        "rq4_summary": rq4,
        "all_points": _df_records(
            agg_df[["hidden_neurons", "activation", "solver", "alpha", "max_iter", "test_accuracy", "train_energy_j", "inf_energy_j", "train_energy_iqr", "inf_energy_iqr", *extra_cols]]
        ),
    })


@app.get("/api/exp3/{dataset}")
def get_exp3(dataset: str) -> Dict[str, Any]:
    att_df = _read_csv_safe(RESULTS_DIR / f"exp3_attainment_{dataset}.csv")
    hist_df = _read_csv_safe(RESULTS_DIR / f"exp3_nsga2_history_{dataset}.csv")
    grid_pf = _read_csv_safe(RESULTS_DIR / f"exp1_pareto_train_{dataset}.csv")
    if att_df.empty:
        raise HTTPException(status_code=404, detail=f"Experiment 3 NSGA-II data missing for {dataset}")

    summary = _read_summary()
    meta = summary.get("nsga2_experiments", {}).get(dataset, {})
    payback = summary.get("carbon_payback", {}).get(dataset, {})

    return _clean_json_val({
        "dataset": dataset,
        "meta": meta,
        "attainment": _df_records(att_df),
        "grid_pareto": _df_records(grid_pf),
        "nsga2_sample_history": _df_records(hist_df.head(300)),
        "carbon_payback": payback,
    })


@app.get("/api/cross-model/{dataset}")
def get_cross_model(dataset: str) -> Dict[str, Any]:
    cm_df = _read_csv_safe(RESULTS_DIR / f"cross_model_{dataset}.csv")
    if cm_df.empty:
        raise HTTPException(status_code=404, detail=f"Cross-model benchmark missing for {dataset}")
    pf_train = extract_pareto_front(cm_df, acc_col="test_accuracy", energy_col="train_energy_j")
    pf_inf = extract_pareto_front(cm_df, acc_col="test_accuracy", energy_col="inf_energy_j")
    knee_train = find_knee_point(pf_train, acc_col="test_accuracy", energy_col="train_energy_j", co2_col="train_co2_g")
    knee_inf = find_knee_point(pf_inf, acc_col="test_accuracy", energy_col="inf_energy_j", co2_col="inf_co2_g")
    summary = _read_summary()
    meta = summary.get("cross_model_benchmarks", {}).get(dataset, {})
    return _clean_json_val({
        "dataset": dataset,
        "meta": meta,
        "models": _df_records(cm_df),
        "pareto_train": _df_records(pf_train),
        "pareto_inf": _df_records(pf_inf),
        "knee_train": knee_train.__dict__ if knee_train else None,
        "knee_inf": knee_inf.__dict__ if knee_inf else None,
    })


def _load_bundle(dataset: str) -> Dict[str, Any]:
    if dataset not in _BUNDLE_CACHE:
        path = MODELS_DIR / f"{dataset}_pareto_surface.joblib"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Hot-swap bundle not found for {dataset}")
        _BUNDLE_CACHE[dataset] = joblib.load(path)
    return _BUNDLE_CACHE[dataset]


@app.get("/api/hotswap/surface/{dataset}")
def get_hotswap_surface(dataset: str) -> Dict[str, Any]:
    bundle = _load_bundle(dataset)
    return _clean_json_val({
        "dataset": dataset,
        "entries": bundle["entries"],
        "n_features": len(bundle.get("feature_names", [])),
    })


@app.post("/api/hotswap/predict")
def post_hotswap_predict(req: HotSwapPredictRequest) -> Dict[str, Any]:
    bundle = _load_bundle(req.dataset)
    entries = bundle["entries"]
    models = bundle["models"]

    selected = select_hotswap_model(entries, battery_pct=req.battery_pct, min_accuracy=req.min_accuracy)
    active_model = models[selected["model_key"]]

    ds = load_dataset(req.dataset, mortgage_max_rows=bundle.get("mortgage_max_rows"))
    profiler = _get_profiler()

    def _run_batch() -> np.ndarray:
        preds = None
        for _ in range(req.batch_repeats):
            preds = active_model.predict(ds.X_test)
        return preds

    y_pred, meas = profiler.measure_callable(_run_batch)
    scale_1k = 1000.0 / max(1, req.batch_repeats)
    row_idx = req.sample_idx % len(ds.X_test)
    pred_single = int(active_model.predict(ds.X_test[row_idx : row_idx + 1])[0])
    true_single = int(ds.y_test[row_idx])

    return _clean_json_val({
        "selected": selected,
        "live_test_accuracy": float(np.mean(y_pred == ds.y_test)),
        "batch_latency_ms": (meas.wall_time_s / req.batch_repeats) * 1000.0,
        "measured_energy_1k_j": meas.cpu_energy_j * scale_1k,
        "measured_co2_1k_mg": meas.co2_grams_total * scale_1k * 1000.0,
        "sample_idx": row_idx,
        "sample_total": len(ds.X_test),
        "predicted_class": pred_single,
        "ground_truth_class": true_single,
        "is_correct": bool(pred_single == true_single),
    })


@app.post("/api/live-experiment")
def post_live_experiment(req: LiveExperimentRequest) -> Dict[str, Any]:
    """Run an on-demand side-by-side model comparison with real EnergyProfiler telemetry."""
    if req.dataset not in {"iris", "glass", "ionosphere", "diabetes", "telco_churn", "titanic"}:
        raise HTTPException(status_code=400, detail="Live demo supports iris, glass, ionosphere, diabetes, telco_churn, titanic")
    if req.mlp_solver not in {"sgd", "adam", "lbfgs"}:
        raise HTTPException(status_code=400, detail="Invalid solver")
    if req.mlp_activation not in {"logistic", "tanh", "relu"}:
        raise HTTPException(status_code=400, detail="Invalid activation")

    ds = load_dataset(req.dataset)
    profiler = _get_profiler()
    candidates = [
        ("LogisticRegression", LogisticRegression(max_iter=500, random_state=RANDOM_SEED)),
        ("DecisionTree", DecisionTreeClassifier(max_depth=6, random_state=RANDOM_SEED)),
        ("RandomForest (n=50)", RandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED, n_jobs=1)),
        (
            f"MLP (h={req.mlp_hidden}, {req.mlp_activation}, {req.mlp_solver})",
            MLPClassifier(
                hidden_layer_sizes=(req.mlp_hidden,),
                activation=req.mlp_activation,
                solver=req.mlp_solver,
                max_iter=200,
                random_state=RANDOM_SEED,
            ),
        ),
    ]

    rows: List[Dict[str, Any]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for label, clf in candidates:
            _, tr_m = profiler.measure_callable(lambda c=clf: c.fit(ds.X_train, ds.y_train))

            def _inf(c=clf):
                out = None
                for _ in range(100):
                    out = c.predict(ds.X_test)
                return out

            y_pred, inf_m = profiler.measure_callable(_inf)
            avg = "binary" if ds.n_classes == 2 else "weighted"
            rows.append({
                "model": label,
                "test_accuracy": float(accuracy_score(ds.y_test, y_pred)),
                "f1_score": float(f1_score(ds.y_test, y_pred, average=avg, zero_division=0)),
                "train_time_ms": tr_m.wall_time_s * 1000.0,
                "train_energy_j": tr_m.cpu_energy_j,
                "train_co2_mg": tr_m.co2_grams_total * 1000.0,
                "inf_energy_1k_j": inf_m.cpu_energy_j * 10.0,
                "inf_co2_1k_mg": inf_m.co2_grams_total * 10.0 * 1000.0,
            })

    return _clean_json_val({"dataset": req.dataset, "n_train": len(ds.X_train), "n_test": len(ds.X_test), "results": rows})
