"""Authentic dataset loading and paper-faithful preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml, load_iris
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

from src.config import DATA_DIR, MORTGAGE_DROP_COLS, RANDOM_SEED, TEST_SIZE


@dataclass
class PreparedDataset:
    key: str
    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    n_classes: int
    raw_rows: int
    processed_rows: int
    raw_attributes: int
    encoded_features: int
    source_note: str


def _ensure_uci_csvs() -> None:
    """Download UCI/OpenML CSVs into data/ if missing (requires network if not cached)."""
    iris_path = DATA_DIR / "iris.csv"
    if not iris_path.exists():
        iris = load_iris(as_frame=True)
        iris.frame.to_csv(iris_path, index=False)

    diabetes_path = DATA_DIR / "diabetes.csv"
    if not diabetes_path.exists():
        X, y = fetch_openml(data_id=37, return_X_y=True, as_frame=True)
        pd.concat([X, y.rename("Outcome")], axis=1).to_csv(diabetes_path, index=False)

    glass_path = DATA_DIR / "glass.csv"
    if not glass_path.exists():
        from ucimlrepo import fetch_ucirepo

        glass = fetch_ucirepo(id=42)
        pd.concat([glass.data.features, glass.data.targets], axis=1).to_csv(glass_path, index=False)

    iono_path = DATA_DIR / "ionosphere.csv"
    if not iono_path.exists():
        from ucimlrepo import fetch_ucirepo

        iono = fetch_ucirepo(id=52)
        pd.concat([iono.data.features, iono.data.targets], axis=1).to_csv(iono_path, index=False)


def _split_and_scale(
    key: str,
    name: str,
    X_df: pd.DataFrame,
    y_raw: pd.Series,
    num_cols: List[str],
    cat_cols: List[str],
    raw_rows: int,
    raw_attributes: int,
    source_note: str,
    max_rows: Optional[int] = None,
) -> PreparedDataset:
    """Apply stratified sampling (optional), 75:25 split, and train-only preprocessing/scaling."""
    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))
    processed_rows = len(X_df)

    if max_rows is not None and len(X_df) > max_rows:
        X_df, _, y, _ = train_test_split(
            X_df,
            y,
            train_size=max_rows,
            stratify=y,
            random_state=RANDOM_SEED,
        )
        X_df = X_df.reset_index(drop=True)

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    transformers = []
    if num_cols:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", MinMaxScaler(feature_range=(0.0, 1.0))),
            ]
        )
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", cat_pipe, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    X_train = preprocessor.fit_transform(X_train_df).astype(np.float64)
    X_test = preprocessor.transform(X_test_df).astype(np.float64)

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [f"f_{i}" for i in range(X_train.shape[1])]

    return PreparedDataset(
        key=key,
        name=name,
        X_train=X_train,
        X_test=X_test,
        y_train=np.asarray(y_train, dtype=np.int64),
        y_test=np.asarray(y_test, dtype=np.int64),
        feature_names=feature_names,
        n_classes=int(len(np.unique(y))),
        raw_rows=raw_rows,
        processed_rows=processed_rows,
        raw_attributes=raw_attributes,
        encoded_features=int(X_train.shape[1]),
        source_note=source_note,
    )


def load_dataset(key: str, mortgage_max_rows: Optional[int] = None) -> PreparedDataset:
    """Load and preprocess any supported dataset with strict train-only scaling and 75:25 split."""
    _ensure_uci_csvs()
    key = key.lower().strip()

    if key == "iris":
        df = pd.read_csv(DATA_DIR / "iris.csv")
        X_df = df.drop(columns=["target"])
        return _split_and_scale(
            key="iris",
            name="Iris",
            X_df=X_df,
            y_raw=df["target"],
            num_cols=list(X_df.columns),
            cat_cols=[],
            raw_rows=len(df),
            raw_attributes=X_df.shape[1],
            source_note="UCI Iris (150 instances, 4 attributes)",
        )

    if key == "glass":
        df = pd.read_csv(DATA_DIR / "glass.csv")
        drop_cols = [c for c in ["Id", "id", "Type_of_glass"] if c in df.columns]
        X_df = df.drop(columns=drop_cols)
        return _split_and_scale(
            key="glass",
            name="Glass Identification",
            X_df=X_df,
            y_raw=df["Type_of_glass"],
            num_cols=list(X_df.columns),
            cat_cols=[],
            raw_rows=len(df),
            raw_attributes=X_df.shape[1],
            source_note="UCI Glass Identification (214 instances, 9 chemical/RI attributes)",
        )

    if key == "ionosphere":
        df = pd.read_csv(DATA_DIR / "ionosphere.csv")
        X_df = df.drop(columns=["Class"])
        return _split_and_scale(
            key="ionosphere",
            name="Ionosphere",
            X_df=X_df,
            y_raw=df["Class"],
            num_cols=list(X_df.columns),
            cat_cols=[],
            raw_rows=len(df),
            raw_attributes=X_df.shape[1],
            source_note="UCI Ionosphere (351 instances, 34 attributes)",
        )

    if key == "diabetes":
        df = pd.read_csv(DATA_DIR / "diabetes.csv")
        X_df = df.drop(columns=["Outcome"])
        return _split_and_scale(
            key="diabetes",
            name="Pima Indians Diabetes",
            X_df=X_df,
            y_raw=df["Outcome"],
            num_cols=list(X_df.columns),
            cat_cols=[],
            raw_rows=len(df),
            raw_attributes=X_df.shape[1],
            source_note="UCI / OpenML #37 Pima Indians Diabetes (768 instances, 8 attributes)",
        )

    if key == "mortgage":
        vals_path = DATA_DIR / "mortgage" / "train_values.csv"
        labels_path = DATA_DIR / "mortgage" / "train_labels.csv"
        if not vals_path.exists() or not labels_path.exists():
            raise FileNotFoundError("Mortgage CSVs missing under data/mortgage/")
        vals = pd.read_csv(vals_path)
        labels = pd.read_csv(labels_path)
        df = vals.merge(labels, on="row_id", how="inner")
        raw_rows = len(df)

        # Paper Section III exact preprocessing:
        # 1) Filter upper outliers (keep instances where loan_amount<=3750, applicant_income<=2000, population<=35000)
        mask = (
            (df["loan_amount"] <= 3750)
            & (df["applicant_income"] <= 2000)
            & (df["population"] <= 35000)
        )
        df = df.loc[mask].reset_index(drop=True)

        # 2) Drop the 5 noisy/flat variables
        drop_cols = [c for c in MORTGAGE_DROP_COLS if c in df.columns] + ["accepted"]
        y_raw = df["accepted"]
        X_df = df.drop(columns=drop_cols)

        cat_cols = [
            "loan_type",
            "property_type",
            "loan_purpose",
            "occupancy",
            "preapproval",
            "applicant_ethnicity",
            "applicant_race",
            "applicant_sex",
            "co_applicant",
        ]
        cat_cols = [c for c in cat_cols if c in X_df.columns]
        for c in cat_cols:
            X_df[c] = X_df[c].astype(str)
        num_cols = [c for c in X_df.columns if c not in cat_cols]

        note = (
            f"Kaggle dinu1763/mortgage-loan-approval ({raw_rows:,} raw -> {len(df):,} cleaned; "
            f"17 attributes -> one-hot + [0,1] scaled)"
        )
        if mortgage_max_rows is not None and len(df) > mortgage_max_rows:
            note += f" [stratified {mortgage_max_rows:,}-row sample]"

        return _split_and_scale(
            key="mortgage",
            name="Mortgage Loan Approval",
            X_df=X_df,
            y_raw=y_raw,
            num_cols=num_cols,
            cat_cols=cat_cols,
            raw_rows=raw_rows,
            raw_attributes=X_df.shape[1],
            source_note=note,
            max_rows=mortgage_max_rows,
        )

    if key == "telco_churn":
        path = DATA_DIR / "telco_churn" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        df = pd.read_csv(path)
        raw_rows = len(df)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        y_raw = df["Churn"]
        X_df = df.drop(columns=[c for c in ["customerID", "Churn"] if c in df.columns])
        X_df["SeniorCitizen"] = X_df["SeniorCitizen"].astype(str)
        num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        cat_cols = [c for c in X_df.columns if c not in num_cols]
        return _split_and_scale(
            key="telco_churn",
            name="Telco Customer Churn",
            X_df=X_df,
            y_raw=y_raw,
            num_cols=num_cols,
            cat_cols=cat_cols,
            raw_rows=raw_rows,
            raw_attributes=X_df.shape[1],
            source_note=f"Kaggle blastchar/telco-customer-churn ({raw_rows:,} instances, {X_df.shape[1]} raw attributes)",
        )

    if key == "titanic":
        path = DATA_DIR / "titanic" / "Titanic-Dataset.csv"
        df = pd.read_csv(path)
        raw_rows = len(df)
        y_raw = df["Survived"]
        keep_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
        X_df = df[keep_cols].copy()
        X_df["Pclass"] = X_df["Pclass"].astype(str)
        num_cols = ["Age", "SibSp", "Parch", "Fare"]
        cat_cols = ["Pclass", "Sex", "Embarked"]
        return _split_and_scale(
            key="titanic",
            name="Titanic Survival",
            X_df=X_df,
            y_raw=y_raw,
            num_cols=num_cols,
            cat_cols=cat_cols,
            raw_rows=raw_rows,
            raw_attributes=X_df.shape[1],
            source_note=f"Kaggle yasserh/titanic-dataset ({raw_rows:,} instances, {X_df.shape[1]} core attributes)",
        )

    raise ValueError(f"Unknown dataset key: {key}")
