"""Configuration constants matching Brownlee et al. (GI 2021) and Proposal Topic 1."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"

for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TEST_SIZE = 0.25
CV_FOLDS = 5
INFERENCE_REPEATS = 1000
DEFAULT_INTERLEAVED_REPEATS = 30

# Paper Table I: MLP Hyperparameters for UCI datasets (8 * 3 * 3 * 5 * 4 = 1,440 configs)
UCI_MLP_GRID: Dict[str, List] = {
    "hidden_layer_sizes": [(2,), (5,), (10,), (20,), (50,), (100,), (200,), (500,)],
    "activation": ["logistic", "tanh", "relu"],
    "solver": ["sgd", "adam", "lbfgs"],
    "alpha": [0.002, 0.001, 0.0005, 0.0002, 0.0001],
    "max_iter": [100, 200, 400, 800],
}

# Paper Table I: Restricted MLP Hyperparameters for large Mortgage dataset (3 * 3 * 2 * 2 * 2 = 72 configs)
MORTGAGE_MLP_GRID: Dict[str, List] = {
    "hidden_layer_sizes": [(50,), (100,), (200,)],
    "activation": ["logistic", "tanh", "relu"],
    "solver": ["sgd", "adam"],
    "alpha": [0.0005, 0.0001],
    "max_iter": [200, 400],
}

# Paper Section VI: NSGA-II settings in jMetalPy
NSGA2_CONFIG = {
    "population_size": 8,
    "max_evaluations": 240,  # 1/6th of 1,440 grid configs
    "crossover_probability": 1.0,
    "crossover_distribution_index": 20.0,
    "mutation_probability": 1.0 / 5.0,  # 1/n for 5 decision variables
    "mutation_distribution_index": 20.0,
    "hidden_min": 2,
    "hidden_max": 500,
    "alpha_min": 0.0001,
    "alpha_max": 0.002,
    "max_iter_min": 100,
    "max_iter_max": 800,
    "activations": ["logistic", "tanh", "relu"],
    "solvers": ["sgd", "adam", "lbfgs"],
}

# Paper Section III: Columns dropped and outlier thresholds for Mortgage dataset
MORTGAGE_DROP_COLS = ["row_id", "lender", "msa_md", "state_code", "county_code"]


@dataclass(frozen=True)
class DatasetMeta:
    key: str
    display_name: str
    source: str
    role: str  # "paper_uci", "paper_kaggle", "proposal_kaggle"


DATASETS_META: Dict[str, DatasetMeta] = {
    "iris": DatasetMeta("iris", "Iris", "UCI / sklearn", "paper_uci"),
    "glass": DatasetMeta("glass", "Glass Identification", "UCI / OpenML", "paper_uci"),
    "ionosphere": DatasetMeta("ionosphere", "Ionosphere", "UCI / OpenML", "paper_uci"),
    "diabetes": DatasetMeta("diabetes", "Pima Indians Diabetes", "UCI / Kaggle / OpenML", "paper_uci"),
    "mortgage": DatasetMeta("mortgage", "Mortgage Loan Approval", "Kaggle (dinu1763/mortgage-loan-approval)", "paper_kaggle"),
    "telco_churn": DatasetMeta("telco_churn", "Telco Customer Churn", "Kaggle (blastchar/telco-customer-churn)", "proposal_kaggle"),
    "titanic": DatasetMeta("titanic", "Titanic Survival", "Kaggle / OpenML", "proposal_kaggle"),
}


def expand_mlp_grid(grid: Dict[str, List]) -> List[Dict]:
    """Expand hyperparameter grid dictionary into a deterministic list of config dicts."""
    keys = list(grid.keys())
    configs = []
    for idx, combo in enumerate(product(*(grid[k] for k in keys))):
        cfg = dict(zip(keys, combo))
        cfg["config_id"] = idx
        configs.append(cfg)
    return configs
