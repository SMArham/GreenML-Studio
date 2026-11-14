"""Comprehensive QA tests for Backend REST API, Frontend responsive assets, and error handling."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from backend.api import (
    FRONTEND_DIR,
    HotSwapPredictRequest,
    LiveExperimentRequest,
    get_cross_model,
    get_exp1,
    get_exp2,
    get_exp3,
    get_hotswap_surface,
    get_overview,
    health,
    list_figures,
    post_hotswap_predict,
    post_live_experiment,
    serve_frontend,
)


def test_frontend_responsive_structure_and_assets() -> None:
    resp = serve_frontend()
    html_path = Path(resp.path)
    assert html_path.exists()
    html = html_path.read_text(encoding="utf-8")

    # Semantic navbar, hamburger toggle, footer, and mobile viewport required for QA
    assert 'name="viewport"' in html
    assert 'id="menu-toggle"' in html
    assert '<nav class="nav-tabs"' in html
    assert '<footer class="site-footer">' in html
    assert "fig-deck-container" in html
    assert "All Essential" in html
    assert "exp1-flops-chart" in html
    assert "exp2-section5-kpis" in html

    css = (FRONTEND_DIR / "styles.css").read_text(encoding="utf-8")
    assert "@media (max-width: 768px)" in css
    assert "@media (max-width: 1024px)" in css
    assert ".menu-toggle" in css

    js = (FRONTEND_DIR / "app.js").read_text(encoding="utf-8")
    assert "resizeVisibleCharts" in js
    assert "loadFiguresGallery" in js


def test_overview_health_and_figures_api() -> None:
    from backend.api import app

    # Swagger / ReDoc docs disabled per minimalist requirements
    assert app.docs_url is None
    assert app.redoc_url is None
    assert app.openapi_url is None

    h = health()
    assert h["status"] == "ok"
    assert len(h["paper_datasets"]) == 5

    ov = get_overview()
    assert "hardware" in ov
    assert "paper_datasets" in ov

    figs = list_figures()
    assert len(figs["figures"]) >= 5
    assert "items" in figs
    assert len(figs["items"]) >= 5


def test_exp1_exp2_enriched_metrics() -> None:
    e1 = get_exp1("diabetes")
    assert len(e1["configs"]) == 144
    sample_cfg = e1["configs"][0]
    assert sample_cfg["param_count"] > 0
    assert sample_cfg["inf_flops_per_sample"] > 0
    # Pareto configs must have 5-fold CV populated for generalization audit
    assert all(row["cv_accuracy"] is not None for row in e1["pareto_train"])

    e2 = get_exp2("diabetes")
    assert len(e2["spearman"]) > 0
    assert len(e2["categorical"]) > 0
    rq4 = e2["rq4_summary"]
    assert "spearman_flops_vs_inf_energy" in rq4
    assert "compact_long_best_acc" in rq4


def test_exp3_api_iris() -> None:
    e3 = get_exp3("iris")
    assert len(e3["attainment"]) > 0
    assert len(e3["grid_pareto"]) > 0
    assert e3["meta"]["max_evaluations"] == 240


def test_api_error_handling_validation() -> None:
    with pytest.raises(HTTPException) as exc_404:
        get_exp1("nonexistent_dataset")
    assert exc_404.value.status_code == 404

    with pytest.raises(HTTPException) as exc_bad_ds:
        post_live_experiment(LiveExperimentRequest(dataset="bad_ds"))
    assert exc_bad_ds.value.status_code == 400

    with pytest.raises(HTTPException) as exc_bad_sol:
        post_live_experiment(LiveExperimentRequest(dataset="iris", mlp_solver="bad_solver"))
    assert exc_bad_sol.value.status_code == 400


def test_live_experiment_endpoint() -> None:
    req = LiveExperimentRequest(dataset="iris", mlp_hidden=20, mlp_solver="adam", mlp_activation="relu")
    data = post_live_experiment(req)
    assert len(data["results"]) == 4
    for row in data["results"]:
        assert 0.0 <= row["test_accuracy"] <= 1.0
        assert row["train_energy_j"] > 0.0
        assert row["inf_energy_1k_j"] > 0.0


def test_cross_model_and_hotswap_endpoints() -> None:
    cm = get_cross_model("diabetes")
    assert len(cm["models"]) >= 30
    assert cm["knee_train"] is not None

    surf = get_hotswap_surface("diabetes")
    assert len(surf["entries"]) >= 2

    low_bat = post_hotswap_predict(HotSwapPredictRequest(dataset="diabetes", battery_pct=15.0, min_accuracy=0.0, sample_idx=0, batch_repeats=20))
    high_bat = post_hotswap_predict(HotSwapPredictRequest(dataset="diabetes", battery_pct=100.0, min_accuracy=0.0, sample_idx=0, batch_repeats=20))
    assert low_bat["selected"]["inf_energy_1k_j"] <= high_bat["selected"]["inf_energy_1k_j"]
    assert 0.0 <= low_bat["live_test_accuracy"] <= 1.0
    assert low_bat["batch_latency_ms"] > 0.0

