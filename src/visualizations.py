"""Publication-grade Matplotlib PNG figures (Paper Figs 1-9 + Cross-Model & Carbon Payback)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR
from src.pareto import KneeSummary

sns.set_theme(style="whitegrid", context="notebook")


# ---------------------------------------------------------------------------
# Static Matplotlib Figures (saved to outputs/figures/)
# ---------------------------------------------------------------------------

def save_fig1_normality_histogram(raw_30rep: pd.DataFrame, audit_summary: pd.DataFrame, dataset_key: str) -> Path:
    """Reproduce Paper Fig 1: Histogram of 30 repeated CPU energy runs showing non-normality."""
    first_cid = int(audit_summary["config_id"].iloc[0])
    sub = raw_30rep[raw_30rep["config_id"] == first_cid]
    row = audit_summary.iloc[0]

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=160)
    ax.hist(sub["train_energy_j"], bins=10, color="#2E7D32", edgecolor="white", alpha=0.85)
    ax.axvline(row["median_j"], color="#D84315", linestyle="--", linewidth=2, label=f"Median = {row['median_j']:.3f} J")
    ax.set_title(f"Fig 1: 30-Repeat CPU Training Energy Distribution ({dataset_key.title()} | {row['label']})")
    ax.set_xlabel("CPU Energy (Joules)")
    ax.set_ylabel("Frequency (out of 30 interleaved runs)")
    ax.legend()
    ax.text(
        0.97,
        0.88,
        f"Skewness = {row['skewness']:.2f}\nShapiro-Wilk p = {row['shapiro_p']:.4f}\nIQR = {row['iqr_j']:.4f} J",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="#F5F5F5", edgecolor="#BDBDBD"),
        fontsize=9,
    )
    out_path = FIGURES_DIR / f"fig1_normality_histogram_{dataset_key}.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def save_fig2_3_time_vs_energy(agg_df: pd.DataFrame, dataset_key: str) -> Path:
    """Reproduce Paper Figs 2 & 3: CPU Time vs CPU Energy for Training and Inference."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=160)

    # Training
    ax = axes[0]
    ax.scatter(agg_df["train_cpu_time_s"], agg_df["train_energy_j"], s=20, alpha=0.65, color="#1565C0")
    med_t = float(agg_df["train_cpu_time_s"].median())
    short_df = agg_df[agg_df["train_cpu_time_s"] <= med_t]
    long_df = agg_df[agg_df["train_cpu_time_s"] > med_t]
    if len(short_df) >= 2:
        xs = np.linspace(short_df["train_cpu_time_s"].min(), short_df["train_cpu_time_s"].max(), 30)
        p_s = np.poly1d(np.polyfit(short_df["train_cpu_time_s"], short_df["train_energy_j"], 1))
        ax.plot(xs, p_s(xs), color="#E65100", linewidth=2, label="Short-run slope")
    if len(long_df) >= 2:
        xl = np.linspace(long_df["train_cpu_time_s"].min(), long_df["train_cpu_time_s"].max(), 30)
        p_l = np.poly1d(np.polyfit(long_df["train_cpu_time_s"], long_df["train_energy_j"], 1))
        ax.plot(xl, p_l(xl), color="#2E7D32", linestyle="--", linewidth=2, label="Long-run slope")
    ax.set_title(f"Fig 2: CPU Time vs Energy (Training - {dataset_key.title()})")
    ax.set_xlabel("CPU Time (seconds)")
    ax.set_ylabel("CPU Energy (Joules)")
    ax.legend()

    # Inference
    ax = axes[1]
    ax.scatter(agg_df["inf_cpu_time_s"], agg_df["inf_energy_j"], s=20, alpha=0.65, color="#00838F")
    ax.set_title(f"Fig 3: CPU Time vs Energy (Inference 1k - {dataset_key.title()})")
    ax.set_xlabel("CPU Time (seconds for 1k runs)")
    ax.set_ylabel("CPU Energy (Joules for 1k runs)")

    out_path = FIGURES_DIR / f"fig2_3_time_vs_energy_{dataset_key}.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def save_fig4_grid_spread(agg_df: pd.DataFrame, dataset_key: str) -> Path:
    """Reproduce Paper Fig 4: Full grid accuracy vs training energy colored by solver/neurons."""
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    palette = {"sgd": "#1E88E5", "adam": "#43A047", "lbfgs": "#E53935"}
    for sol, grp in agg_df.groupby("solver"):
        ax.errorbar(
            grp["test_accuracy"],
            grp["train_energy_j"],
            yerr=grp["train_energy_iqr"] / 2.0,
            fmt="o",
            markersize=5,
            alpha=0.65,
            color=palette.get(str(sol), "#546E7A"),
            ecolor=palette.get(str(sol), "#546E7A"),
            elinewidth=0.7,
            label=f"solver={sol}",
        )
    ax.set_title(f"Fig 4: All MLP Grid Configs — Test Accuracy vs Training CPU Energy ({dataset_key.title()})")
    ax.set_xlabel("Accuracy on Unseen Test Set")
    ax.set_ylabel("Training CPU Energy (Joules)")
    ax.legend(title="Solver")
    out_path = FIGURES_DIR / f"fig4_grid_spread_{dataset_key}.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def save_fig5_6_pareto_knee(
    pareto_train: pd.DataFrame,
    pareto_inf: pd.DataFrame,
    knee_train: Optional[KneeSummary],
    knee_inf: Optional[KneeSummary],
    dataset_key: str,
) -> Path:
    """Reproduce Paper Figs 5 & 6: Training & Inference Pareto fronts with Knee points."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=160)

    for ax, pf, knee, e_col, title in [
        (axes[0], pareto_train, knee_train, "train_energy_j", f"Fig 5: Training Pareto Front ({dataset_key.title()})"),
        (axes[1], pareto_inf, knee_inf, "inf_energy_j", f"Fig 6: Inference 1k Pareto Front ({dataset_key.title()})"),
    ]:
        if pf.empty:
            continue
        ax.plot(pf["test_accuracy"], pf[e_col], "-o", color="#1E88E5", markersize=7, linewidth=1.8, label="Pareto Front")
        if knee is not None:
            kr = knee.knee_row
            mr = knee.max_accuracy_row
            ax.scatter([kr["test_accuracy"]], [kr[e_col]], s=170, color="#2E7D32", marker="*", zorder=5, label="Knee Point")
            ax.scatter([mr["test_accuracy"]], [mr[e_col]], s=90, color="#D32F2F", marker="D", zorder=5, label="Max Accuracy")
            ax.annotate(
                f"Knee: -{knee.energy_saved_pct:.1f}% energy\n(-{knee.accuracy_drop_pct_points:.2f}% acc)",
                xy=(kr["test_accuracy"], kr[e_col]),
                xytext=(0.05, 0.78),
                textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="#2E7D32"),
                bbox=dict(boxstyle="round", facecolor="#E8F5E9", edgecolor="#2E7D32"),
                fontsize=9,
            )
        ax.set_title(title)
        ax.set_xlabel("Accuracy on Unseen Test Set")
        ax.set_ylabel("CPU Energy (Joules)")
        ax.legend(loc="upper left")

    out_path = FIGURES_DIR / f"fig5_6_pareto_knee_{dataset_key}.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def save_fig7_hyperparam_matrix(pareto_df: pd.DataFrame, dataset_key: str, stage: str = "train") -> Path:
    """Reproduce Paper Fig 7: Hyperparameter trends sorted along the Pareto front."""
    e_col = "train_energy_j" if stage == "train" else "inf_energy_j"
    pf = pareto_df.sort_values(by=[e_col, "test_accuracy"]).reset_index(drop=True)
    if pf.empty:
        return FIGURES_DIR / f"fig7_empty_{dataset_key}.png"

    fig, ax = plt.subplots(figsize=(10.5, max(3.5, 0.42 * len(pf) + 1.4)), dpi=160)
    ax.axis("off")

    cols = ["activation", "alpha", "hidden_neurons", "max_iter", "solver", e_col, "test_accuracy"]
    col_labels = ["Activation", "Alpha", "Hidden Size", "Max Iter", "Solver", "CPU Energy (J)", "Test Acc"]

    cell_text = []
    cell_colors = []
    act_colors = {"logistic": "#BBDEFB", "tanh": "#FFE0B2", "relu": "#C8E6C9"}
    sol_colors = {"sgd": "#FFF9C4", "adam": "#B2EBF2", "lbfgs": "#F8BBD0"}

    max_e = max(float(pf[e_col].max()), 1e-9)
    max_h = max(float(pf["hidden_neurons"].max()), 1.0)
    max_it = max(float(pf["max_iter"].max()), 1.0)

    for _, r in pf.iterrows():
        cell_text.append([
            str(r["activation"]),
            f"{r['alpha']:.4f}",
            f"{int(r['hidden_neurons'])}",
            f"{int(r['max_iter'])}",
            str(r["solver"]),
            f"{r[e_col]:.3f}",
            f"{r['test_accuracy']*100:.2f}%",
        ])
        e_norm = float(r[e_col]) / max_e
        h_norm = float(r["hidden_neurons"]) / max_h
        it_norm = float(r["max_iter"]) / max_it
        cell_colors.append([
            act_colors.get(str(r["activation"]), "#FFFFFF"),
            "#F5F5F5",
            plt.cm.Blues(0.15 + 0.65 * h_norm),
            plt.cm.Purples(0.15 + 0.65 * it_norm),
            sol_colors.get(str(r["solver"]), "#FFFFFF"),
            plt.cm.Reds(0.10 + 0.65 * e_norm),
            plt.cm.Greens(0.20 + 0.65 * float(r["test_accuracy"])),
        ])

    tbl = ax.table(cellText=cell_text, colLabels=col_labels, cellColours=cell_colors, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.35)
    ax.set_title(f"Fig 7: Hyperparameter Trends Along Pareto Front ({dataset_key.title()} — {stage.title()})", pad=12)

    out_path = FIGURES_DIR / f"fig7_hyperparam_trends_{dataset_key}_{stage}.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def save_fig8_train_vs_inf(agg_df: pd.DataFrame, dataset_key: str) -> Path:
    """Reproduce Paper Fig 8: Training CPU Energy vs Inference CPU Energy with IQR bars (RQ4)."""
    fig, ax = plt.subplots(figsize=(7.8, 5.2), dpi=160)
    ax.errorbar(
        agg_df["train_energy_j"],
        agg_df["inf_energy_j"],
        xerr=agg_df["train_energy_iqr"] / 2.0,
        yerr=agg_df["inf_energy_iqr"] / 2.0,
        fmt="o",
        markersize=4,
        alpha=0.65,
        color="#37474F",
        ecolor="#90A4AE",
        elinewidth=0.6,
    )
    ax.set_title(f"Fig 8: Training vs Inference (1k runs) CPU Energy ({dataset_key.title()})")
    ax.set_xlabel("CPU Energy for Training (Joules)")
    ax.set_ylabel("CPU Energy for Testing / Inference 1k (Joules)")
    out_path = FIGURES_DIR / f"fig8_train_vs_inf_{dataset_key}.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def save_fig9_nsga2_attainment(
    grid_pareto: pd.DataFrame,
    attainment_df: pd.DataFrame,
    dataset_key: str,
) -> Path:
    """Reproduce Paper Fig 9: Grid Pareto front vs NSGA-II min/median/max attainment curves."""
    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=160)
    if not attainment_df.empty:
        ax.fill_between(
            attainment_df["accuracy"],
            attainment_df["best_energy_j"],
            attainment_df["worst_energy_j"],
            color="#90CAF9",
            alpha=0.45,
            label="NSGA-II Min-Max Attainment Band",
        )
        ax.plot(
            attainment_df["accuracy"],
            attainment_df["median_energy_j"],
            color="#0D47A1",
            linewidth=2.2,
            label="NSGA-II Median Attainment (240 evals)",
        )
    if not grid_pareto.empty:
        ax.scatter(
            grid_pareto["test_accuracy"],
            grid_pareto["train_energy_j"],
            color="#D32F2F",
            s=55,
            zorder=5,
            label="Exhaustive Grid Pareto Front",
        )
    ax.set_title(f"Fig 9: NSGA-II (240 evals) vs Grid Search Pareto Front ({dataset_key.title()})")
    ax.set_xlabel("Accuracy on Unseen Test Set")
    ax.set_ylabel("Training CPU Energy (Joules)")
    ax.legend()
    out_path = FIGURES_DIR / f"fig9_nsga2_attainment_{dataset_key}.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def save_fig10_cross_model_tradeoff(cross_df: pd.DataFrame, pareto_train: pd.DataFrame, dataset_key: str) -> Path:
    """Extended Plot: Accuracy vs CO2 & Energy across 6 ML Classifier Families (Proposal Topic 1 + Future Work)."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=160)

    ax = axes[0]
    sns.scatterplot(
        data=cross_df,
        x="test_accuracy",
        y="train_energy_j",
        hue="family",
        style="family",
        s=75,
        alpha=0.85,
        ax=ax,
    )
    if not pareto_train.empty:
        ax.plot(pareto_train["test_accuracy"], pareto_train["train_energy_j"], "k--", linewidth=1.5, label="Global Pareto")
    ax.set_title(f"Cross-Paradigm Training Trade-Off ({dataset_key.title()})")
    ax.set_xlabel("Test Accuracy")
    ax.set_ylabel("Training CPU Energy (Joules)")

    ax = axes[1]
    best_per_family = cross_df.sort_values("test_accuracy", ascending=False).groupby("family", as_index=False).first()
    x = np.arange(len(best_per_family))
    ax2 = ax.twinx()
    b1 = ax.bar(x - 0.18, best_per_family["test_accuracy"] * 100, width=0.35, color="#2E7D32", label="Best Accuracy (%)")
    b2 = ax2.bar(x + 0.18, best_per_family["train_co2_g"] * 1000, width=0.35, color="#EF6C00", label="CO2 (mg)")
    ax.set_xticks(x)
    ax.set_xticklabels(best_per_family["family"], rotation=20, ha="right")
    ax.set_ylabel("Best Test Accuracy (%)", color="#2E7D32")
    ax2.set_ylabel("Training CO2 Emissions (milligrams)", color="#EF6C00")
    ax.set_title(f"Best Accuracy vs Carbon Footprint by Classifier ({dataset_key.title()})")

    out_path = FIGURES_DIR / f"fig10_cross_model_{dataset_key}.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def save_fig11_carbon_payback(payback_info: Dict[str, float], dataset_key: str) -> Path:
    """Second-order plot: Cumulative Net Energy Saved vs Production Inference Count."""
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=160)
    saved_per_sample_j = payback_info["saved_per_sample_uj"] / 1e6
    if saved_per_sample_j <= 0 or not np.isfinite(payback_info["breakeven_inferences_nsga2"]):
        ax.text(0.5, 0.5, "No positive per-inference saving found on this dataset", ha="center", va="center")
    else:
        max_n = max(payback_info["breakeven_inferences_grid"] * 1.4, 1e5)
        n_vals = np.linspace(0, max_n, 200)
        net_grid = n_vals * saved_per_sample_j - payback_info["grid_search_energy_j"]
        net_nsga = n_vals * saved_per_sample_j - payback_info["nsga2_search_energy_j"]
        ax.plot(n_vals / 1e6, net_grid, color="#D32F2F", linewidth=2, label="Net Energy: Grid Search")
        ax.plot(n_vals / 1e6, net_nsga, color="#2E7D32", linewidth=2.2, label="Net Energy: NSGA-II Search")
        ax.axhline(0, color="#424242", linestyle="--", linewidth=1)
        ax.set_xlabel("Production Inferences (Millions)")
        ax.set_ylabel("Net Energy Saved (Joules)")
        ax.set_title(f"Carbon Payback Break-Even Curve ({dataset_key.title()})")
        ax.legend()
    out_path = FIGURES_DIR / f"fig11_carbon_payback_{dataset_key}.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
