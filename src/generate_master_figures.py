"""Generate Curated Essential Publication Figures for GreenML Studio.

Generates:
1. 01_master_research_synthesis.png (Comprehensive 4-Panel Executive Research Poster)
2. 02_pareto_frontiers_multi_dataset.png (4-Dataset Pareto & Knee Comparison)
3. 03_hyperparameter_drivers_decoupling.png (Spearman Drivers + Train vs Inference)
4. 04_nsga2_payback_synthesis.png (NSGA-II Attainment + Carbon Payback Horizon)
5. 05_cross_model_benchmark_synthesis.png (Cross-Paradigm Classifier Families)
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, RESULTS_DIR

# Set high-quality typography & Apple-inspired styling
plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]
plt.rcParams["axes.edgecolor"] = "#d2d2d7"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.color"] = "#e5e5ea"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.7

APPLE_BLUE = "#0071e3"
APPLE_GREEN = "#34c759"
APPLE_ORANGE = "#ff9500"
APPLE_PURPLE = "#5856d6"
APPLE_RED = "#ff3b30"
APPLE_CHARCOAL = "#1d1d1f"
APPLE_GRAY = "#86868b"
APPLE_BG = "#fbfbfd"


def create_master_synthesis():
    """Create the 4-panel executive research synthesis figure."""
    fig = plt.figure(figsize=(18, 12), dpi=180, facecolor="#ffffff")
    fig.suptitle(
        "GreenML Studio: Empirical Accuracy vs. Energy Trade-Off in Machine Learning\n"
        "Multi-Objective Pareto Optimization, Hyperparameter Drivers & Carbon Payback Horizons",
        fontsize=16,
        fontweight="bold",
        color=APPLE_CHARCOAL,
        y=0.97,
    )

    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22, left=0.06, right=0.96, top=0.90, bottom=0.07)

    # ----------------------------------------------------
    # PANEL A: Multi-Dataset Pareto Frontiers & Knee Points
    # ----------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(APPLE_BG)
    datasets = [("diabetes", "Diabetes", APPLE_BLUE), ("iris", "Iris", APPLE_GREEN), 
                ("ionosphere", "Ionosphere", APPLE_PURPLE), ("glass", "Glass", APPLE_ORANGE)]

    for ds_key, label, color in datasets:
        agg_file = RESULTS_DIR / f"exp1_agg_{ds_key}.csv"
        pareto_file = RESULTS_DIR / f"exp1_pareto_train_{ds_key}.csv"
        if agg_file.exists():
            df = pd.read_csv(agg_file)
            ax1.scatter(df["test_accuracy"] * 100, df["train_energy_j"], s=18, alpha=0.35, color=color, label=f"{label} (All Configs)")
        if pareto_file.exists():
            pdf = pd.read_csv(pareto_file).sort_values("test_accuracy")
            ax1.plot(pdf["test_accuracy"] * 100, pdf["train_energy_j"], color=color, linewidth=2.2, label=f"{label} Pareto Front")

    # Mark Knee Point for Diabetes
    ax1.scatter([74.48], [0.2529], color=APPLE_GREEN, s=180, marker="*", edgecolor="#ffffff", linewidth=1.5, zorder=10, label="Diabetes Knee (-85.7% J)")
    ax1.annotate(
        "Knee: -85.7% Energy\n(Only -1.04% Acc Drop)",
        xy=(74.48, 0.2529),
        xytext=(58, 1.1),
        arrowprops=dict(facecolor=APPLE_GREEN, edgecolor=APPLE_GREEN, arrowstyle="->", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor=APPLE_GREEN, lw=1.2),
        fontsize=9,
        fontweight="bold",
        color=APPLE_CHARCOAL,
    )

    ax1.set_title("Panel A: Multi-Dataset Pareto Frontiers & Recommended Knee Points", fontsize=12, fontweight="bold", color=APPLE_CHARCOAL, pad=10)
    ax1.set_xlabel("Test Accuracy (%)", fontsize=10, fontweight="bold", color=APPLE_CHARCOAL)
    ax1.set_ylabel("Training CPU Energy (Joules)", fontsize=10, fontweight="bold", color=APPLE_CHARCOAL)
    ax1.set_yscale("log")
    ax1.grid(True)
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # ----------------------------------------------------
    # PANEL B: Cross-Model Architecture Benchmark
    # ----------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(APPLE_BG)
    cm_file = RESULTS_DIR / "cross_model_telco_churn.csv"
    if cm_file.exists():
        cm_df = pd.read_csv(cm_file)
        fam_colors = {
            "LogisticRegression": APPLE_BLUE,
            "DecisionTree": APPLE_GREEN,
            "RandomForest": "#00838f",
            "SVM": APPLE_PURPLE,
            "KNN": APPLE_ORANGE,
            "ExtraTrees": "#d81b60",
        }
        for fam, grp in cm_df.groupby("family"):
            ax2.scatter(
                grp["test_accuracy"] * 100,
                grp["train_energy_j"],
                s=grp["f1_score"] * 80,
                color=fam_colors.get(fam, APPLE_GRAY),
                alpha=0.85,
                edgecolor="#ffffff",
                linewidth=0.8,
                label=fam,
            )

        # Annotate Key Observations
        ax2.annotate(
            "KNN (k=21): Knee Point\n96.4% Energy Saved\n(3.3% Acc Trade-off)",
            xy=(77.40, 0.0036),
            xytext=(70.5, 0.08),
            arrowprops=dict(facecolor=APPLE_ORANGE, edgecolor=APPLE_ORANGE, arrowstyle="->", lw=1.5),
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor=APPLE_ORANGE, lw=1.2),
            fontsize=9,
            fontweight="bold",
            color=APPLE_CHARCOAL,
        )

        ax2.annotate(
            "Logistic Regression: Peak Accuracy\n(80.7% Acc | 0.098 J)",
            xy=(80.69, 0.0987),
            xytext=(74.5, 0.8),
            arrowprops=dict(facecolor=APPLE_BLUE, edgecolor=APPLE_BLUE, arrowstyle="->", lw=1.5),
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor=APPLE_BLUE, lw=1.2),
            fontsize=9,
            fontweight="bold",
            color=APPLE_CHARCOAL,
        )

    ax2.set_title("Panel B: Cross-Model Classifier Families (Telco Churn Benchmark)", fontsize=12, fontweight="bold", color=APPLE_CHARCOAL, pad=10)
    ax2.set_xlabel("Test Accuracy (%)", fontsize=10, fontweight="bold", color=APPLE_CHARCOAL)
    ax2.set_ylabel("Training CPU Energy (Joules)", fontsize=10, fontweight="bold", color=APPLE_CHARCOAL)
    ax2.set_yscale("log")
    ax2.grid(True)
    ax2.legend(loc="lower left", fontsize=8, framealpha=0.9)

    # ----------------------------------------------------
    # PANEL C: Train vs Inference Decoupling (RQ4)
    # ----------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(APPLE_BG)
    diab_agg = RESULTS_DIR / "exp1_agg_diabetes.csv"
    if diab_agg.exists():
        d_df = pd.read_csv(diab_agg)
        sc = ax3.scatter(
            d_df["train_energy_j"],
            d_df["inf_energy_j"],
            c=d_df["hidden_neurons"],
            cmap="viridis",
            s=36,
            alpha=0.75,
            edgecolor="#ffffff",
            linewidth=0.6,
        )
        cbar = plt.colorbar(sc, ax=ax3, pad=0.02)
        cbar.set_label("Hidden Neurons", fontsize=9, color=APPLE_CHARCOAL)
        cbar.ax.tick_params(labelsize=8)

        # Draw median dividing lines for quadrants
        med_train = d_df["train_energy_j"].median()
        med_inf = d_df["inf_energy_j"].median()
        ax3.axvline(med_train, color=APPLE_GRAY, linestyle=":", alpha=0.7)
        ax3.axhline(med_inf, color=APPLE_GRAY, linestyle=":", alpha=0.7)

        ax3.text(
            0.05, 0.92,
            "High-Train / Low-Inference Sweet Spot:\nTrain once (long iterations) -> Ultra-green in production",
            transform=ax3.transAxes,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f5e9", edgecolor=APPLE_GREEN, lw=1.2),
            fontsize=8.5,
            color=APPLE_CHARCOAL,
        )

    ax3.set_title("Panel C: Training vs. Inference Stage Decoupling (RQ4 | Diabetes)", fontsize=12, fontweight="bold", color=APPLE_CHARCOAL, pad=10)
    ax3.set_xlabel("Training Energy (Joules)", fontsize=10, fontweight="bold", color=APPLE_CHARCOAL)
    ax3.set_ylabel("Inference 1k Energy (Joules)", fontsize=10, fontweight="bold", color=APPLE_CHARCOAL)
    ax3.grid(True)

    # ----------------------------------------------------
    # PANEL D: NSGA-II Attainment & Carbon Payback Horizon
    # ----------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(APPLE_BG)

    # Carbon Payback Simulation lines
    pb_daily = 100_000
    grid_search_j = 2023.53
    nsga_search_j = 343.99
    saved_per_sample_j = 9.766e-6

    inferences_m = np.linspace(0, 250, 200)
    net_saved_nsga = inferences_m * 1e6 * saved_per_sample_j - nsga_search_j
    net_saved_grid = inferences_m * 1e6 * saved_per_sample_j - grid_search_j

    ax4.plot(inferences_m, net_saved_nsga, color=APPLE_GREEN, linewidth=2.5, label="NSGA-II Strategy (Break-Even: 35.2M inf / 352 days)")
    ax4.plot(inferences_m, net_saved_grid, color=APPLE_RED, linewidth=2.0, linestyle="--", label="Brute Grid Strategy (Break-Even: 207.2M inf / 2,072 days)")
    ax4.axhline(0, color=APPLE_GRAY, linestyle="-", linewidth=1.0, alpha=0.6)

    ax4.annotate(
        "NSGA-II Break-Even:\n352 Days at 100k inf/day\n(83% Search Compute Saved)",
        xy=(35.22, 0),
        xytext=(50, -600),
        arrowprops=dict(facecolor=APPLE_GREEN, edgecolor=APPLE_GREEN, arrowstyle="->", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor=APPLE_GREEN, lw=1.2),
        fontsize=9,
        fontweight="bold",
        color=APPLE_CHARCOAL,
    )

    ax4.set_title("Panel D: Carbon Payback Break-Even Horizon (Diabetes Benchmark)", fontsize=12, fontweight="bold", color=APPLE_CHARCOAL, pad=10)
    ax4.set_xlabel("Production Inferences (Millions)", fontsize=10, fontweight="bold", color=APPLE_CHARCOAL)
    ax4.set_ylabel("Net Energy Saved (Joules)", fontsize=10, fontweight="bold", color=APPLE_CHARCOAL)
    ax4.grid(True)
    ax4.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    out_file = FIGURES_DIR / "01_master_research_synthesis.png"
    fig.savefig(out_file, dpi=180)
    plt.close(fig)
    print(f"Created: {out_file}")
    return out_file


def create_individual_essentials():
    """Create 4 focused individual essential figures."""
    
    # 1. 02_pareto_frontiers_multi_dataset.png
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=160, facecolor="#ffffff")
    fig.suptitle("Core Visual 1: Empirical Pareto Frontiers & Knee Point Analysis Across 4 Benchmarks", fontsize=14, fontweight="bold", y=0.98)
    
    datasets = [
        ("diabetes", "Pima Indians Diabetes (UCI)", axes[0, 0], APPLE_BLUE, 74.48, 0.2529, "85.7% Saved (-1.04% Acc)"),
        ("iris", "Iris Plant Recognition (UCI)", axes[0, 1], APPLE_GREEN, 94.74, 0.052, "75.6% Saved (-2.63% Acc)"),
        ("glass", "Glass Identification (UCI)", axes[1, 0], APPLE_ORANGE, 71.30, 0.125, "54.2% Saved (-3.70% Acc)"),
        ("ionosphere", "Ionosphere Radar Returns (UCI)", axes[1, 1], APPLE_PURPLE, 92.61, 0.210, "51.5% Saved (-1.70% Acc)"),
    ]
    
    for ds_key, title, ax, color, knee_x, knee_y, knee_text in datasets:
        ax.set_facecolor(APPLE_BG)
        agg_file = RESULTS_DIR / f"exp1_agg_{ds_key}.csv"
        pareto_file = RESULTS_DIR / f"exp1_pareto_train_{ds_key}.csv"
        if agg_file.exists():
            df = pd.read_csv(agg_file)
            ax.scatter(df["test_accuracy"] * 100, df["train_energy_j"], s=22, alpha=0.4, color=color, label="Grid Models")
        if pareto_file.exists():
            pdf = pd.read_csv(pareto_file).sort_values("test_accuracy")
            ax.plot(pdf["test_accuracy"] * 100, pdf["train_energy_j"], color=APPLE_CHARCOAL, linewidth=2.2, label="Pareto Front")
            ax.scatter([knee_x], [knee_y], s=160, marker="*", color=APPLE_GREEN, edgecolor="#ffffff", linewidth=1.2, zorder=5, label="Knee Point")
            ax.text(0.04, 0.88, f"Knee: {knee_text}", transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor="#ffffff", edgecolor=APPLE_GREEN), fontsize=8.5, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Test Accuracy (%)", fontsize=9.5)
        ax.set_ylabel("Training Energy (J)", fontsize=9.5)
        ax.set_yscale("log")
        ax.grid(True)
        ax.legend(fontsize=8, loc="lower left")
    
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_pareto_frontiers_multi_dataset.png", dpi=160)
    plt.close(fig)
    print("Created: 02_pareto_frontiers_multi_dataset.png")

    # 2. 03_hyperparameter_drivers_decoupling.png
    fig, ax1 = plt.subplots(figsize=(11, 6.2), dpi=160, facecolor="#ffffff")
    fig.suptitle("Core Visual 2: Hyperparameter Attribution Drivers (Spearman Rank Correlation)", fontsize=13, fontweight="bold", y=0.97)
    
    ax1.set_facecolor(APPLE_BG)
    spearman_file = RESULTS_DIR / "exp2_spearman_diabetes.csv"
    if spearman_file.exists():
        sdf = pd.read_csv(spearman_file)
        y_pos = np.arange(len(sdf))
        bar_colors = [APPLE_BLUE if r > 0 else APPLE_ORANGE for r in sdf["spearman_rho"]]
        bars = ax1.barh(y_pos, sdf["spearman_rho"], color=bar_colors, alpha=0.88, height=0.55)
        ax1.set_yticks(y_pos)
        
        target_map = {
            "inf_energy_j": "Inference Energy (J)",
            "train_energy_j": "Training Energy (J)",
            "test_accuracy": "Test Accuracy",
        }
        labels = []
        for _, r in sdf.iterrows():
            param = r["parameter"].replace("_", " ").title()
            target = target_map.get(r["target"], r["target"].replace("_", " ").title())
            labels.append(f"{param}  →  {target}")
        ax1.set_yticklabels(labels, fontsize=9.5, fontweight="500")
        
        ax1.axvline(0, color=APPLE_CHARCOAL, linewidth=0.9)
        ax1.set_title("Spearman Rank Correlation: Architecture Hyperparameters vs Targets (Diabetes Benchmark)", fontsize=11, fontweight="bold", pad=12)
        ax1.set_xlabel("Spearman Rank Correlation Coefficient (ρ)", fontsize=10, labelpad=8)
        ax1.set_xlim(-0.75, 0.98)
        ax1.grid(True, linestyle="--", alpha=0.5)
        
        for i, bar in enumerate(bars):
            val = sdf.iloc[i]["spearman_rho"]
            pval = sdf.iloc[i].get("p_value", None)
            offset = 0.02 if val >= 0 else -0.02
            ha = "left" if val >= 0 else "right"
            p_text = f" (p < 0.001)" if (pval is not None and pval < 0.001) else (f" (p={pval:.3f})" if pval is not None else "")
            ax1.text(val + offset, bar.get_y() + bar.get_height() / 2, f"ρ = {val:+.3f}{p_text}", va="center", ha=ha, fontsize=8.5, fontweight="bold", color=APPLE_CHARCOAL)
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURES_DIR / "03_hyperparameter_drivers_decoupling.png", dpi=160)
    plt.close(fig)
    print("Created: 03_hyperparameter_drivers_decoupling.png")

    # 3. 04_nsga2_payback_synthesis.png
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=160, facecolor="#ffffff")
    fig.suptitle("Core Visual 3: NSGA-II Multi-Objective Optimization & Production Payback Horizons", fontsize=14, fontweight="bold", y=0.98)
    
    ax1.set_facecolor(APPLE_BG)
    att_file = RESULTS_DIR / "exp3_attainment_diabetes.csv"
    pareto_file = RESULTS_DIR / "exp1_pareto_train_diabetes.csv"
    if att_file.exists():
        att_df = pd.read_csv(att_file).sort_values("accuracy")
        ax1.fill_between(att_df["accuracy"] * 100, att_df["best_energy_j"], att_df["worst_energy_j"], color=APPLE_BLUE, alpha=0.2, label="NSGA-II 3-Run Attainment Band")
        ax1.plot(att_df["accuracy"] * 100, att_df["median_energy_j"], color=APPLE_BLUE, linewidth=2.5, label="NSGA-II Median Attainment")
    if pareto_file.exists():
        pdf = pd.read_csv(pareto_file).sort_values("test_accuracy")
        ax1.scatter(pdf["test_accuracy"] * 100, pdf["train_energy_j"], color=APPLE_RED, s=60, marker="o", label="Exhaustive Grid Pareto Points")
    ax1.set_title("NSGA-II (48 evals) vs Full Grid (1,440 configs)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Test Accuracy (%)", fontsize=9.5)
    ax1.set_ylabel("Training Energy (Joules)", fontsize=9.5)
    ax1.set_yscale("log")
    ax1.grid(True)
    ax1.legend(loc="upper left", fontsize=8.5)

    ax2.set_facecolor(APPLE_BG)
    inferences_m = np.linspace(0, 250, 200)
    saved_per_sample_j = 9.766e-6
    net_saved_nsga = inferences_m * 1e6 * saved_per_sample_j - 343.99
    net_saved_grid = inferences_m * 1e6 * saved_per_sample_j - 2023.53
    ax2.plot(inferences_m, net_saved_nsga, color=APPLE_GREEN, linewidth=2.5, label="NSGA-II Payback (352 days)")
    ax2.plot(inferences_m, net_saved_grid, color=APPLE_RED, linewidth=2.0, linestyle="--", label="Grid Payback (2,072 days)")
    ax2.axhline(0, color=APPLE_GRAY, linestyle="-", linewidth=1.0)
    ax2.set_title("Carbon Payback Break-Even Horizon at 100k Queries/Day", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Production Inferences (Millions)", fontsize=9.5)
    ax2.set_ylabel("Net Energy Saved (Joules)", fontsize=9.5)
    ax2.grid(True)
    ax2.legend(loc="upper left", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_nsga2_payback_synthesis.png", dpi=160)
    plt.close(fig)
    print("Created: 04_nsga2_payback_synthesis.png")

    # 4. 05_cross_model_benchmark_synthesis.png
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=160, facecolor="#ffffff")
    fig.suptitle("Core Visual 4: Cross-Paradigm Classifier Families (Telco Churn & Mortgage)", fontsize=14, fontweight="bold", y=0.98)
    
    fam_colors = {
        "LogisticRegression": APPLE_BLUE,
        "DecisionTree": APPLE_GREEN,
        "RandomForest": "#00838f",
        "SVM": APPLE_PURPLE,
        "KNN": APPLE_ORANGE,
        "ExtraTrees": "#d81b60",
    }
    
    for ax, ds, title in [(ax1, "telco_churn", "Telco Customer Churn (Kaggle)"), (ax2, "mortgage", "Mortgage Loan Approval (Kaggle)")]:
        ax.set_facecolor(APPLE_BG)
        cm_file = RESULTS_DIR / f"cross_model_{ds}.csv"
        if cm_file.exists():
            cm_df = pd.read_csv(cm_file)
            for fam, grp in cm_df.groupby("family"):
                ax.scatter(grp["test_accuracy"] * 100, grp["train_energy_j"], s=grp["f1_score"] * 70, color=fam_colors.get(fam, APPLE_GRAY), label=fam, alpha=0.85)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Test Accuracy (%)", fontsize=9.5)
        ax.set_ylabel("Training Energy (Joules)", fontsize=9.5)
        ax.set_yscale("log")
        ax.grid(True)
        ax.legend(loc="lower left", fontsize=8)
    
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_cross_model_benchmark_synthesis.png", dpi=160)
    plt.close(fig)
    print("Created: 05_cross_model_benchmark_synthesis.png")


if __name__ == "__main__":
    create_master_synthesis()
    create_individual_essentials()
