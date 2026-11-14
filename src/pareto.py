"""Pareto front extraction, Knee-point detection, Hypervolume, and Attainment curves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class KneeSummary:
    has_clear_knee: bool
    knee_row: Dict
    max_accuracy_row: Dict
    min_energy_row: Dict
    accuracy_drop_pct_points: float
    energy_saved_pct: float
    co2_saved_pct: float


def extract_pareto_front(
    df: pd.DataFrame,
    acc_col: str = "test_accuracy",
    energy_col: str = "train_energy_j",
) -> pd.DataFrame:
    """Return non-dominated rows maximizing accuracy and minimizing energy.

    A row B is dominated if there exists another row A with:
    A.accuracy >= B.accuracy AND A.energy <= B.energy AND strictly better in at least one.
    """
    if df.empty:
        return df.copy()

    work = df.dropna(subset=[acc_col, energy_col]).copy().reset_index(drop=True)
    acc = work[acc_col].to_numpy(dtype=float)
    eng = work[energy_col].to_numpy(dtype=float)

    n = len(work)
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_dominated[i]:
            continue
        # Any point j that is >= acc[i] and <= eng[i] with strict improvement in either
        dominates_i = (acc >= acc[i]) & (eng <= eng[i]) & ((acc > acc[i]) | (eng < eng[i]))
        if np.any(dominates_i):
            is_dominated[i] = True

    pareto = work.loc[~is_dominated].copy()
    pareto = pareto.sort_values(by=[energy_col, acc_col], ascending=[True, True]).reset_index(drop=True)
    return pareto


def find_knee_point(
    pareto_df: pd.DataFrame,
    acc_col: str = "test_accuracy",
    energy_col: str = "train_energy_j",
    co2_col: Optional[str] = None,
    max_acc_drop_tol: float = 0.12,
) -> Optional[KneeSummary]:
    """Identify the viable knee point on a 2D Pareto front (maximize accuracy, minimize energy).

    Restricts knee selection to the economically viable high-accuracy region
    (within max_acc_drop_tol of peak accuracy when >=3 such points exist) so
    collapsed low-accuracy extremes are never misidentified as knee points.
    """
    if pareto_df.empty:
        return None

    pf_all = pareto_df.sort_values(by=[energy_col, acc_col], ascending=[True, True]).reset_index(drop=True)
    min_energy_row = pf_all.iloc[0].to_dict()
    max_acc_val = float(pf_all[acc_col].max())
    max_acc_idx = int(pf_all[acc_col].idxmax())
    max_accuracy_row = pf_all.iloc[max_acc_idx].to_dict()

    # Restrict knee geometry to viable upper-accuracy segment when enough points exist
    viable = pf_all[pf_all[acc_col] >= max_acc_val - max_acc_drop_tol].reset_index(drop=True)
    pf = viable if len(viable) >= 3 else pf_all

    if len(pf) < 3 or np.isclose(pf[acc_col].max(), pf[acc_col].min()) or np.isclose(pf[energy_col].max(), pf[energy_col].min()):
        knee_row = min_energy_row if len(pf) == 1 else pf.iloc[0].to_dict()
        max_e = float(max_accuracy_row[energy_col])
        knee_e = float(knee_row[energy_col])
        e_saved = ((max_e - knee_e) / max_e * 100.0) if max_e > 0 else 0.0
        c_saved = 0.0
        if co2_col and co2_col in pf_all.columns and float(max_accuracy_row.get(co2_col, 0.0)) > 0:
            c_saved = (float(max_accuracy_row[co2_col]) - float(knee_row[co2_col])) / float(max_accuracy_row[co2_col]) * 100.0
        return KneeSummary(
            has_clear_knee=False,
            knee_row=knee_row,
            max_accuracy_row=max_accuracy_row,
            min_energy_row=min_energy_row,
            accuracy_drop_pct_points=(float(max_accuracy_row[acc_col]) - float(knee_row[acc_col])) * 100.0,
            energy_saved_pct=e_saved,
            co2_saved_pct=c_saved,
        )

    # Normalize to [0, 1] where x = accuracy (higher better), y = energy (lower better)
    x = (pf[acc_col].to_numpy(dtype=float) - pf[acc_col].min()) / (pf[acc_col].max() - pf[acc_col].min())
    y = (pf[energy_col].to_numpy(dtype=float) - pf[energy_col].min()) / (pf[energy_col].max() - pf[energy_col].min())

    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    chord = p2 - p1
    chord_len = np.linalg.norm(chord)

    if chord_len < 1e-12:
        knee_idx = 0
    else:
        pts = np.column_stack([x, y])
        diff = pts - p1
        signed_dist = (chord[0] * diff[:, 1] - chord[1] * diff[:, 0]) / chord_len
        best_idx = int(np.argmin(signed_dist))
        if signed_dist[best_idx] >= -1e-6:
            utopia_dist = np.sqrt((1.0 - x) ** 2 + (0.0 - y) ** 2)
            knee_idx = int(np.argmin(utopia_dist))
        else:
            knee_idx = best_idx

    knee_row = pf.iloc[knee_idx].to_dict()
    max_e = float(max_accuracy_row[energy_col])
    knee_e = float(knee_row[energy_col])
    e_saved = ((max_e - knee_e) / max_e * 100.0) if max_e > 0 else 0.0
    c_saved = 0.0
    if co2_col and co2_col in pf.columns and float(max_accuracy_row.get(co2_col, 0.0)) > 0:
        c_saved = (float(max_accuracy_row[co2_col]) - float(knee_row[co2_col])) / float(max_accuracy_row[co2_col]) * 100.0

    return KneeSummary(
        has_clear_knee=(e_saved > 5.0 and len(pf) >= 3),
        knee_row=knee_row,
        max_accuracy_row=max_accuracy_row,
        min_energy_row=min_energy_row,
        accuracy_drop_pct_points=(float(max_accuracy_row[acc_col]) - float(knee_row[acc_col])) * 100.0,
        energy_saved_pct=e_saved,
        co2_saved_pct=c_saved,
    )


def compute_hypervolume_2d(
    pareto_df: pd.DataFrame,
    acc_col: str = "test_accuracy",
    energy_col: str = "train_energy_j",
    ref_error: float = 1.0,
    ref_energy: Optional[float] = None,
) -> float:
    """Compute exact 2D hypervolume in minimized (error = 1 - accuracy, energy) space."""
    if pareto_df.empty:
        return 0.0
    pf = extract_pareto_front(pareto_df, acc_col=acc_col, energy_col=energy_col)
    if pf.empty:
        return 0.0

    errors = 1.0 - pf[acc_col].to_numpy(dtype=float)
    energies = pf[energy_col].to_numpy(dtype=float)

    if ref_energy is None:
        ref_energy = float(np.max(energies) * 1.1)

    # Filter points strictly inside reference box and sort ascending by error
    valid = (errors < ref_error) & (energies < ref_energy)
    if not np.any(valid):
        return 0.0

    pts = sorted(zip(errors[valid], energies[valid]), key=lambda t: (t[0], t[1]))
    hv = 0.0
    prev_energy = ref_energy
    for err, eng in pts:
        if eng < prev_energy:
            hv += (ref_error - err) * (prev_energy - eng)
            prev_energy = eng
    return float(hv)


def compute_attainment_curves(
    runs: List[pd.DataFrame],
    acc_col: str = "test_accuracy",
    energy_col: str = "train_energy_j",
    n_grid: int = 120,
) -> pd.DataFrame:
    """Compute min (best), median, and max (worst) empirical attainment curves across runs.

    Matches Paper Fig 9: for each accuracy threshold a, reports the minimum energy
    achieved with accuracy >= a across runs (lower energy = better).
    """
    if not runs:
        return pd.DataFrame()

    all_acc = np.concatenate([r[acc_col].to_numpy(dtype=float) for r in runs if not r.empty])
    if len(all_acc) == 0:
        return pd.DataFrame()

    acc_min = float(np.min(all_acc))
    acc_max = float(np.max(all_acc))
    acc_grid = np.linspace(acc_min, acc_max, n_grid)

    mat = np.full((len(runs), n_grid), np.nan, dtype=float)
    for r_idx, run_df in enumerate(runs):
        if run_df.empty:
            continue
        r_acc = run_df[acc_col].to_numpy(dtype=float)
        r_eng = run_df[energy_col].to_numpy(dtype=float)
        for g_idx, a_thresh in enumerate(acc_grid):
            feasible = r_eng[r_acc >= a_thresh - 1e-12]
            if len(feasible) > 0:
                mat[r_idx, g_idx] = np.min(feasible)

    # Only keep accuracy levels attained by at least one run (and check all-runs coverage too)
    valid_any = np.any(~np.isnan(mat), axis=0)
    acc_valid = acc_grid[valid_any]
    mat_valid = mat[:, valid_any]

    # For missing high-accuracy points in weaker runs, fill with column max or nan-aware quantiles
    best_curve = np.nanmin(mat_valid, axis=0)
    median_curve = np.nanmedian(mat_valid, axis=0)
    worst_curve = np.nanmax(mat_valid, axis=0)

    return pd.DataFrame({
        "accuracy": acc_valid,
        "best_energy_j": best_curve,
        "median_energy_j": median_curve,
        "worst_energy_j": worst_curve,
        "runs_attaining": np.sum(~np.isnan(mat_valid), axis=0),
    })
