"""Automated tests verifying preprocessing integrity, Pareto math, telemetry, and NSGA-II."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import UCI_MLP_GRID, expand_mlp_grid
from src.data_loader import load_dataset
from src.energy_profiler import EnergyProfiler
from src.extended_benchmark import compute_carbon_payback, select_hotswap_model
from src.paper_experiments import _MLPEnergyProblem, _select_balanced_grid, add_mlp_complexity_columns
from src.pareto import compute_attainment_curves, compute_hypervolume_2d, extract_pareto_front, find_knee_point


def test_dataset_preprocessing_no_leakage() -> None:
    for key in ["iris", "glass", "ionosphere", "diabetes", "telco_churn", "titanic"]:
        ds = load_dataset(key)
        assert ds.X_train.shape[0] > ds.X_test.shape[0]
        assert np.isfinite(ds.X_train).all()
        assert np.isfinite(ds.X_test).all()
        # Train features must be strictly within [0, 1] after train-only MinMaxScaler
        assert ds.X_train.min() >= -1e-9
        assert ds.X_train.max() <= 1.0 + 1e-9


def test_mortgage_outlier_filter_and_drops() -> None:
    ds = load_dataset("mortgage", mortgage_max_rows=2000)
    assert ds.processed_rows > 400000
    assert ds.raw_attributes == 17  # 22 raw minus 5 dropped noise columns
    assert np.isfinite(ds.X_train).all()
    assert set(np.unique(ds.y_train)).issubset({0, 1})


def test_pareto_dominance_and_knee_detection() -> None:
    df = pd.DataFrame({
        "config_id": [0, 1, 2, 3, 4],
        "test_accuracy": [0.70, 0.85, 0.932, 0.943, 0.80],
        "train_energy_j": [1.0, 2.0, 4.0, 20.0, 15.0],  # id=4 is dominated; id=2 is clear knee!
        "train_co2_g": [0.001, 0.002, 0.004, 0.020, 0.015],
    })
    pf = extract_pareto_front(df, acc_col="test_accuracy", energy_col="train_energy_j")
    assert set(pf["config_id"].tolist()) == {0, 1, 2, 3}
    knee = find_knee_point(pf, acc_col="test_accuracy", energy_col="train_energy_j", co2_col="train_co2_g")
    assert knee is not None
    assert knee.has_clear_knee
    assert knee.knee_row["config_id"] == 2
    assert np.isclose(knee.energy_saved_pct, 80.0)


def test_hypervolume_and_attainment_curves() -> None:
    r1 = pd.DataFrame({"test_accuracy": [0.80, 0.90], "train_energy_j": [2.0, 5.0]})
    r2 = pd.DataFrame({"test_accuracy": [0.82, 0.92], "train_energy_j": [3.0, 6.0]})
    hv = compute_hypervolume_2d(r1, ref_energy=10.0)
    assert hv > 0.0
    att = compute_attainment_curves([r1, r2], n_grid=10)
    assert not att.empty
    assert (att["best_energy_j"] <= att["median_energy_j"] + 1e-12).all()
    assert (att["median_energy_j"] <= att["worst_energy_j"] + 1e-12).all()


def test_energy_profiler_and_hotswap() -> None:
    profiler = EnergyProfiler(country_iso_code="PAK")
    res, meas = profiler.measure_callable(lambda: sum(i * i for i in range(50000)))
    assert res > 0
    assert meas.wall_time_s > 0
    assert meas.cpu_energy_j > 0
    assert meas.co2_grams_total > 0

    payback = compute_carbon_payback(1000.0, 150.0, 10.0, 2.0, n_test_samples=100)
    assert payback["breakeven_inferences_nsga2"] < payback["breakeven_inferences_grid"]

    entries = [
        {"tier": "Ultra-Green", "inf_energy_1k_j": 1.0, "test_accuracy": 0.80},
        {"tier": "Balanced", "inf_energy_1k_j": 3.0, "test_accuracy": 0.91},
        {"tier": "Max-Acc", "inf_energy_1k_j": 12.0, "test_accuracy": 0.93},
    ]
    assert select_hotswap_model(entries, battery_pct=10.0)["tier"] == "Ultra-Green"
    assert select_hotswap_model(entries, battery_pct=100.0)["tier"] == "Max-Acc"


def test_jmetal_problem_decode() -> None:
    cfg = _MLPEnergyProblem.decode([123.7, 1.4, 2.1, 0.0007, 350.2])
    assert cfg["hidden_layer_sizes"] == (124,)
    assert cfg["activation"] == "tanh"
    assert cfg["solver"] == "lbfgs"
    assert cfg["max_iter"] == 350


def test_balanced_grid_hyperparam_diversity() -> None:
    full = expand_mlp_grid(UCI_MLP_GRID)
    sub = _select_balanced_grid(full, 72)
    assert len(sub) == 72
    assert len({c["hidden_layer_sizes"] for c in sub}) == 8
    assert len({c["activation"] for c in sub}) == 3
    assert len({c["solver"] for c in sub}) == 3
    assert len({c["alpha"] for c in sub}) >= 4
    assert len({c["max_iter"] for c in sub}) == 4


def test_knee_ignores_collapsed_models() -> None:
    # Mimics Glass inference front where 0.07/0.35 collapsed classifiers exist alongside 0.70..0.80 viable region
    df = pd.DataFrame({
        "config_id": [1, 2, 3, 4, 5],
        "test_accuracy": [0.07, 0.35, 0.70, 0.76, 0.80],
        "inf_energy_j": [1.06, 1.07, 1.21, 1.24, 1.32],
    })
    pf = extract_pareto_front(df, acc_col="test_accuracy", energy_col="inf_energy_j")
    knee = find_knee_point(pf, acc_col="test_accuracy", energy_col="inf_energy_j")
    assert knee is not None
    assert knee.knee_row["test_accuracy"] >= 0.70


def test_mlp_complexity_flops_and_params() -> None:
    ds = load_dataset("iris")  # 4 features, 3 classes
    df = pd.DataFrame({"hidden_neurons": [10]})
    out = add_mlp_complexity_columns(ds, df)
    # params = (4*10 + 10) + (10*3 + 3) = 50 + 33 = 83; flops = 2*(4*10 + 10*3) = 140
    assert int(out["param_count"].iloc[0]) == 83
    assert int(out["inf_flops_per_sample"].iloc[0]) == 140


