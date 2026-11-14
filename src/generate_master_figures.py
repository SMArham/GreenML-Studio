"""Generate Curated Essential Publication Figures & Responsive Standalone Panels for GreenML Studio.

Generates:
1. Master Composites:
   - 01_master_research_synthesis.png
   - 02_pareto_frontiers_multi_dataset.png
   - 03_hyperparameter_drivers_decoupling.png
   - 04_nsga2_payback_synthesis.png
   - 05_cross_model_benchmark_synthesis.png

2. Individual High-Res Mobile-Optimized Standalone Panels:
   - 01_panel_a_pareto.png
   - 01_panel_b_cross_model.png
   - 01_panel_c_decoupling.png
   - 01_panel_d_payback.png
   - 02_pareto_diabetes.png
   - 02_pareto_iris.png
   - 02_pareto_glass.png
   - 02_pareto_ionosphere.png
   - 04_nsga2_attainment.png
   - 04_nsga2_payback.png
   - 05_cross_model_telco.png
   - 05_cross_model_mortgage.png
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


def _draw_panel_a(ax):
    """Draw Panel A: Multi-Dataset Pareto Frontiers & Knee Points."""
    ax.set_facecolor(APPLE_BG)
    datasets = [
        ("diabetes", "Diabetes", APPLE_BLUE),
        ("iris", "Iris", APPLE_GREEN),
        ("ionosphere", "Ionosphere", APPLE_PURPLE),
        ("glass", "Glass", APPLE_ORANGE),
    ]

    for ds_key, label, color in datasets:
        agg_file = RESULTS_DIR / f"exp1_agg_{ds_key}.csv"
        pareto_file = RESULTS_DIR / f"exp1_pareto_train_{ds_key}.csv"
        if agg_file.exists():
            df = pd.read_csv(agg_file)
            ax.scatter(df["test_accuracy"] * 100, df["train_energy_j"], s=18, alpha=0.35, color=color, label=f"{label} (All Configs)")
        if pareto_file.exists():
            pdf = pd.read_csv(pareto_file).sort_values("test_accuracy")
            ax.plot(pdf["test_accuracy"] * 100, pdf["train_energy_j"], color=color, linewidth=2.2, label=f"{label} Pareto Front")

    ax.scatter([74.48], [0.2529], color=APPLE_GREEN, s=180, marker="*", edgecolor="#ffffff", linewidth=1.5, zorder=10, label="Diabetes Knee (-85.7% J)")
    ax.annotate(
        "Knee: -85.7% Energy\n(Only -1.04% Acc Drop)",
        xy=(74.48, 0.2529),
        xytext=(58, 1.1),
        arrowprops=dict(facecolor=APPLE_GREEN, edgecolor=APPLE_GREEN, arrowstyle="->", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor=APPLE_GREEN, lw=1.2),
        fontsize=9.5,
        fontweight="bold",
        color=APPLE_CHARCOAL,
    )
    ax.set_title("Panel A: Multi-Dataset Pareto Frontiers & Recommended Knee Points", fontsize=12, fontweight="bold", color=APPLE_CHARCOAL, pad=10)
    ax.set_xlabel("Test Accuracy (%)", fontsize=10, fontweight="bold", color=APPLE_CHARCOAL)
    ax.set_ylabel("Training CPU Energy (Joules)", fontsize=10, fontweight="bold", color=APPLE_CHARCOAL)
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)


def _draw_panel_b(ax):
    """Draw Panel B: Cross-Model Architecture Benchmark."""
    ax.set_facecolor(APPLE_BG)
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
            ax.scatter(grp["test_accuracy"] * 100, grp["train_energy_j"], s=grp["f1_score"] * 85, color=fam_colors.get(fam, APPLE_GRAY), label=fam, alpha=0.85)

        ax.annotate(
            "KNN (k=21): Knee Point\n96.4% Energy Saved\n(3.3% Acc Trade-off)",
            xy=(77.40, 0.0036),
            xytext=(70.5, 0.08),
            arrowprops=dict(facecolor=APPLE_ORANGE, edgecolor=APPLE_ORANGE, arrowstyle="->", lw=1.5),
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor=APPLE_ORANGE, lw=1.2),
            fontsize=9.5,
            fontweight="bold",
            color=APPLE_CHARCOAL,
        )

        ax.annotate(
            "Logistic Regression: Peak Accuracy\n(80.7% Acc | 0.098 J)",
            xy=(80.69, 0.0987),
            xytext=(74.5, 0.8),
            arrowprops=dict(facecolor=APPLE_BLUE, edgecolor=APPLE_BLUE, arrowstyle="->", lw=1.5),
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor=APPLE_BLUE, lw=1.2),
            fontsize=9.5,
            fontweight="bold",
            color=APPLE_CHARCOAL,
        )

    ax.set_title("Panel B: Cross-Model Classifier Families (Telco Churn Benchmark)", fontsize=12, fontweight="bold", color=APPLE_CHARCOAL, pad=10)
    ax.set_xlabel("Test Accuracy (%)", fontsize=10, fontweight="bold", color=APPLE_CHARCOAL)
    ax.set_ylabel("Training CPU Energy (Joules)", fontsize=10, fontweight="bold", color=APPLE_CHARCOAL)
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)


def _draw_panel_c(ax):
    """Draw Panel C: Train vs Inference Decoupling (RQ4)."""
    ax.set_facecolor(APPLE_BG)
    diab_agg = RESULTS_DIR / "exp1_agg_diabetes.csv"
    if diab_agg.exists():
        d_df = pd.read_csv(diab_agg)
        sc = ax.scatter(
            d_df["train_energy_j"],
            d_df["inf_energy_j"],
            c=d_df["hidden_neurons"],
            cmap="viridis",
            s=42,
            alpha=0.75,
            edgecolor="#ffffff",
            linewidth=0.6,
        )
        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("Hidden Neurons", fontsize=9, color=APPLE_CHARCOAL)

        ax.annotate(
            "Low-Train / Low-Inf\nOptimal Quadrant (10 models)",
            xy=(0.28, 0.00015),
            xytext=(0.55, 0.00030),
            arrowprops=dict(facecolor=APPLE_BLUE, edgecolor=APPLE_BLUE, arrowstyle="->", lw=1.5),
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor=APPLE_BLUE, lw=1.2),
            fontsize=9.5,
            fontweight="bold",
            color=APPLE_CHARCOAL,
        )

    ax.set_title("Panel C: Training vs. Inference Stage Decoupling (RQ4, Diabetes Benchmark)", fontsize=12, fontweight="bold", color=APPLE_CHARCOAL, pad=10)
    ax.set_xlabel("Training Energy (Joules)", fontsize=10, fontweight="bold", color=APPLE_CHARCOAL)
    ax.set_ylabel("Inference Energy (Joules / Sample)", fontsize=10, fontweight="bold", color=APPLE_CHARCOAL)
    ax.set_yscale("log")
    ax.grid(True)


def _draw_panel_d(ax):
    """Draw Panel D: Carbon Payback Break-Even Horizon."""
    ax.set_facecolor(APPLE_BG)
    grid_search_j = 2023.53
    nsga_search_j = 343.99
    saved_per_sample_j = 9.766e-6

    inferences_m = np.linspace(0, 250, 200)
    net_saved_nsga = inferences_m * 1e6 * saved_per_sample_j - nsga_search_j
    net_saved_grid = inferences_m * 1e6 * saved_per_sample_j - grid_search_j

    ax.plot(inferences_m, net_saved_nsga, color=APPLE_GREEN, linewidth=2.5, label="NSGA-II Strategy (Break-Even: 35.2M inf / 352 days)")
    ax.plot(inferences_m, net_saved_grid, color=APPLE_RED, linewidth=2.0, linestyle="--", label="Brute Grid Strategy (Break-Even: 207.2M inf / 2,072 days)")
    ax.axhline(0, color=APPLE_GRAY, linestyle="-", linewidth=1.0, alpha=0.6)

    ax.annotate(
        "NSGA-II Break-Even:\n352 Days at 100k inf/day\n(83% Search Compute Saved)",
        xy=(35.22, 0),
        xytext=(50, -600),
        arrowprops=dict(facecolor=APPLE_GREEN, edgecolor=APPLE_GREEN, arrowstyle="->", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor=APPLE_GREEN, lw=1.2),
        fontsize=9.5,
        fontweight="bold",
        color=APPLE_CHARCOAL,
    )

    ax.set_title("Panel D: Carbon Payback Break-Even Horizon (Diabetes Benchmark)", fontsize=12, fontweight="bold", color=APPLE_CHARCOAL, pad=10)
    ax.set_xlabel("Production Inferences (Millions)", fontsize=10, fontweight="bold", color=APPLE_CHARCOAL)
    ax.set_ylabel("Net Energy Saved (Joules)", fontsize=10, fontweight="bold", color=APPLE_CHARCOAL)
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)


def create_master_synthesis():
    """Create the 4-panel executive research synthesis figure and standalone panels."""
    # 1. Composite 4-Panel Figure
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
    _draw_panel_a(fig.add_subplot(gs[0, 0]))
    _draw_panel_b(fig.add_subplot(gs[0, 1]))
    _draw_panel_c(fig.add_subplot(gs[1, 0]))
    _draw_panel_d(fig.add_subplot(gs[1, 1]))

    out_file = FIGURES_DIR / "01_master_research_synthesis.png"
    fig.savefig(out_file, dpi=180)
    plt.close(fig)
    print(f"Created composite: {out_file}")

    # 2. Standalone Individual High-Res Panels for Mobile Stacking
    for name, func, fname in [
        ("Panel A", _draw_panel_a, "01_panel_a_pareto.png"),
        ("Panel B", _draw_panel_b, "01_panel_b_cross_model.png"),
        ("Panel C", _draw_panel_c, "01_panel_c_decoupling.png"),
        ("Panel D", _draw_panel_d, "01_panel_d_payback.png"),
    ]:
        sfig, sax = plt.subplots(figsize=(10, 6.2), dpi=160, facecolor="#ffffff")
        func(sax)
        sfig.tight_layout()
        sfig.savefig(FIGURES_DIR / fname, dpi=160)
        plt.close(sfig)
        print(f"Created panel: {fname}")

    return out_file


def create_individual_essentials():
    """Create individual essential figures and their standalone responsive slices."""
    
    # ----------------------------------------------------
    # 1. 02_pareto_frontiers_multi_dataset.png & Single Slices
    # ----------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=160, facecolor="#ffffff")
    fig.suptitle("Core Visual 1: Empirical Pareto Frontiers & Knee Point Analysis Across 4 Benchmarks", fontsize=14, fontweight="bold", y=0.98)
    
    datasets = [
        ("diabetes", "Pima Indians Diabetes (UCI)", axes[0, 0], APPLE_BLUE, 74.48, 0.2529, "85.7% Saved (-1.04% Acc)"),
        ("iris", "Iris Plant Recognition (UCI)", axes[0, 1], APPLE_GREEN, 94.74, 0.052, "75.6% Saved (-2.63% Acc)"),
        ("glass", "Glass Identification (UCI)", axes[1, 0], APPLE_ORANGE, 71.30, 0.125, "54.2% Saved (-3.70% Acc)"),
        ("ionosphere", "Ionosphere Radar Returns (UCI)", axes[1, 1], APPLE_PURPLE, 92.61, 0.210, "51.5% Saved (-1.70% Acc)"),
    ]
    
    for ds_key, title, ax, color, knee_x, knee_y, knee_text in datasets:
        # Draw on composite ax
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

        # Standalone plot for mobile 1-by-1
        sfig, sax = plt.subplots(figsize=(9, 5.8), dpi=160, facecolor="#ffffff")
        sax.set_facecolor(APPLE_BG)
        if agg_file.exists():
            sax.scatter(df["test_accuracy"] * 100, df["train_energy_j"], s=26, alpha=0.45, color=color, label="Grid Models")
        if pareto_file.exists():
            sax.plot(pdf["test_accuracy"] * 100, pdf["train_energy_j"], color=APPLE_CHARCOAL, linewidth=2.4, label="Pareto Front")
            sax.scatter([knee_x], [knee_y], s=180, marker="*", color=APPLE_GREEN, edgecolor="#ffffff", linewidth=1.4, zorder=5, label="Knee Point")
            sax.text(0.04, 0.88, f"Knee: {knee_text}", transform=sax.transAxes, bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor=APPLE_GREEN), fontsize=9, fontweight="bold")
        sax.set_title(f"Pareto Frontier & Knee: {title}", fontsize=12, fontweight="bold")
        sax.set_xlabel("Test Accuracy (%)", fontsize=10, fontweight="bold")
        sax.set_ylabel("Training Energy (J)", fontsize=10, fontweight="bold")
        sax.set_yscale("log")
        sax.grid(True)
        sax.legend(fontsize=9, loc="lower left")
        sfig.tight_layout()
        sfig.savefig(FIGURES_DIR / f"02_pareto_{ds_key}.png", dpi=160)
        plt.close(sfig)
        print(f"Created panel: 02_pareto_{ds_key}.png")
    
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_pareto_frontiers_multi_dataset.png", dpi=160)
    plt.close(fig)
    print("Created composite: 02_pareto_frontiers_multi_dataset.png")

    # ----------------------------------------------------
    # 2. 03_hyperparameter_drivers_decoupling.png (Single Wide)
    # ----------------------------------------------------
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

    # ----------------------------------------------------
    # 3. 04_nsga2_payback_synthesis.png & Slices
    # ----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=160, facecolor="#ffffff")
    fig.suptitle("Core Visual 3: NSGA-II Multi-Objective Optimization & Production Payback Horizons", fontsize=14, fontweight="bold", y=0.98)
    
    att_file = RESULTS_DIR / "exp3_attainment_diabetes.csv"
    pareto_file = RESULTS_DIR / "exp1_pareto_train_diabetes.csv"
    
    def _draw_nsga_att(ax):
        ax.set_facecolor(APPLE_BG)
        if att_file.exists():
            att_df = pd.read_csv(att_file).sort_values("accuracy")
            ax.fill_between(att_df["accuracy"] * 100, att_df["best_energy_j"], att_df["worst_energy_j"], color=APPLE_BLUE, alpha=0.2, label="NSGA-II 3-Run Attainment Band")
            ax.plot(att_df["accuracy"] * 100, att_df["median_energy_j"], color=APPLE_BLUE, linewidth=2.5, label="NSGA-II Median Attainment")
        if pareto_file.exists():
            pdf = pd.read_csv(pareto_file).sort_values("test_accuracy")
            ax.scatter(pdf["test_accuracy"] * 100, pdf["train_energy_j"], color=APPLE_RED, s=60, marker="o", label="Exhaustive Grid Pareto Points")
        ax.set_title("NSGA-II (48 evals) vs Full Grid (1,440 configs)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Test Accuracy (%)", fontsize=9.5)
        ax.set_ylabel("Training Energy (Joules)", fontsize=9.5)
        ax.set_yscale("log")
        ax.grid(True)
        ax.legend(loc="upper left", fontsize=8.5)

    def _draw_nsga_payback(ax):
        ax.set_facecolor(APPLE_BG)
        inferences_m = np.linspace(0, 250, 200)
        saved_per_sample_j = 9.766e-6
        net_saved_nsga = inferences_m * 1e6 * saved_per_sample_j - 343.99
        net_saved_grid = inferences_m * 1e6 * saved_per_sample_j - 2023.53
        ax.plot(inferences_m, net_saved_nsga, color=APPLE_GREEN, linewidth=2.5, label="NSGA-II Payback (352 days)")
        ax.plot(inferences_m, net_saved_grid, color=APPLE_RED, linewidth=2.0, linestyle="--", label="Grid Payback (2,072 days)")
        ax.axhline(0, color=APPLE_GRAY, linestyle="-", linewidth=1.0)
        ax.set_title("Carbon Payback Break-Even Horizon at 100k Queries/Day", fontsize=11, fontweight="bold")
        ax.set_xlabel("Production Inferences (Millions)", fontsize=9.5)
        ax.set_ylabel("Net Energy Saved (Joules)", fontsize=9.5)
        ax.grid(True)
        ax.legend(loc="upper left", fontsize=8.5)

    _draw_nsga_att(ax1)
    _draw_nsga_payback(ax2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_nsga2_payback_synthesis.png", dpi=160)
    plt.close(fig)
    print("Created composite: 04_nsga2_payback_synthesis.png")

    # Slices for 04
    for title, draw_fn, fname in [
        ("NSGA-II Optimization", _draw_nsga_att, "04_nsga2_attainment.png"),
        ("Carbon Payback Horizon", _draw_nsga_payback, "04_nsga2_payback.png"),
    ]:
        sfig, sax = plt.subplots(figsize=(9, 5.8), dpi=160, facecolor="#ffffff")
        draw_fn(sax)
        sfig.tight_layout()
        sfig.savefig(FIGURES_DIR / fname, dpi=160)
        plt.close(sfig)
        print(f"Created panel: {fname}")

    # ----------------------------------------------------
    # 4. 05_cross_model_benchmark_synthesis.png & Slices
    # ----------------------------------------------------
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
    
    def _draw_cm(ax, ds, title_text):
        ax.set_facecolor(APPLE_BG)
        cm_file = RESULTS_DIR / f"cross_model_{ds}.csv"
        if cm_file.exists():
            cm_df = pd.read_csv(cm_file)
            for fam, grp in cm_df.groupby("family"):
                ax.scatter(grp["test_accuracy"] * 100, grp["train_energy_j"], s=grp["f1_score"] * 70, color=fam_colors.get(fam, APPLE_GRAY), label=fam, alpha=0.85)
        ax.set_title(title_text, fontsize=11, fontweight="bold")
        ax.set_xlabel("Test Accuracy (%)", fontsize=9.5)
        ax.set_ylabel("Training Energy (Joules)", fontsize=9.5)
        ax.set_yscale("log")
        ax.grid(True)
        ax.legend(loc="lower left", fontsize=8)

    _draw_cm(ax1, "telco_churn", "Telco Customer Churn (Kaggle)")
    _draw_cm(ax2, "mortgage", "Mortgage Loan Approval (Kaggle)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_cross_model_benchmark_synthesis.png", dpi=160)
    plt.close(fig)
    print("Created composite: 05_cross_model_benchmark_synthesis.png")

    for ds, title, fname in [
        ("telco_churn", "Cross-Model Benchmark: Telco Customer Churn", "05_cross_model_telco.png"),
        ("mortgage", "Cross-Model Benchmark: Mortgage Loan Approval", "05_cross_model_mortgage.png"),
    ]:
        sfig, sax = plt.subplots(figsize=(9, 5.8), dpi=160, facecolor="#ffffff")
        _draw_cm(sax, ds, title)
        sfig.tight_layout()
        sfig.savefig(FIGURES_DIR / fname, dpi=160)
        plt.close(sfig)
        print(f"Created panel: {fname}")


if __name__ == "__main__":
    create_master_synthesis()
    create_individual_essentials()
