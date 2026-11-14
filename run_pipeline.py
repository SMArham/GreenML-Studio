"""End-to-end research pipeline runner reproducing Brownlee et al. (GI 2021) + Proposal Topic 1."""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

from src.config import MODELS_DIR, RESULTS_DIR
from src.data_loader import load_dataset
from src.energy_profiler import EnergyProfiler
from src.extended_benchmark import (
    compute_carbon_payback,
    export_pareto_hotswap_surface,
    run_cross_model_benchmark,
)
from src.paper_experiments import (
    add_mlp_complexity_columns,
    compute_pareto_cv_if_missing,
    run_experiment_1_grid_search,
    run_experiment_2_parameter_analysis,
    run_experiment_3_nsga2,
    run_normality_audit_30_repeats,
)
from src.pareto import compute_hypervolume_2d, extract_pareto_front, find_knee_point
from src.visualizations import (
    save_fig10_cross_model_tradeoff,
    save_fig11_carbon_payback,
    save_fig1_normality_histogram,
    save_fig2_3_time_vs_energy,
    save_fig4_grid_spread,
    save_fig5_6_pareto_knee,
    save_fig7_hyperparam_matrix,
    save_fig8_train_vs_inf,
    save_fig9_nsga2_attainment,
)


def _save_summary(summary: Dict[str, Any], t_start: float) -> None:
    summary["total_pipeline_runtime_s"] = round(time.perf_counter() - t_start, 2)
    (RESULTS_DIR / "master_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _knee_to_dict(knee) -> Dict[str, Any] | None:
    if knee is None:
        return None
    return {
        "has_clear_knee": knee.has_clear_knee,
        "accuracy_drop_pct_points": round(knee.accuracy_drop_pct_points, 4),
        "energy_saved_pct": round(knee.energy_saved_pct, 2),
        "co2_saved_pct": round(knee.co2_saved_pct, 2),
        "knee_row": knee.knee_row,
        "max_accuracy_row": knee.max_accuracy_row,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GreenML Paper + Proposal Pipeline")
    parser.add_argument(
        "--mode",
        choices=["standard", "paper-exact"],
        default="standard",
        help="standard: balanced factorial grid (~4-6 min); paper-exact: full 1440x30 workstation mode",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse already-completed valid CSV results instead of recomputing from scratch",
    )
    args = parser.parse_args()

    t_start = time.perf_counter()
    profiler = EnergyProfiler(country_iso_code="PAK")
    print(f"[Hardware] {profiler.profile.cpu_model} | TDP={profiler.profile.cpu_tdp_watts}W | Mode={profiler.profile.tracking_mode}", flush=True)

    if args.mode == "paper-exact":
        ds_grid_cfg = {
            "iris": (None, 30, True, None),
            "glass": (None, 30, True, None),
            "ionosphere": (None, 30, True, None),
            "diabetes": (None, 30, True, None),
            "mortgage": (None, 30, True, None),
        }
        inf_repeats = 1000
        nsga_specs = [("iris", 10, 240, 30), ("diabetes", 10, 240, 30), ("ionosphere", 10, 240, 30)]
        cross_repeats = 10
        mortgage_cross_rows = None
    else:
        ds_grid_cfg = {
            "iris": (144, 3, True, None),
            "glass": (48, 2, False, None),
            "ionosphere": (48, 2, False, None),
            "diabetes": (144, 3, False, None),
            "mortgage": (18, 2, False, 1000),
        }
        inf_repeats = 200  # Normalized to 1,000 test applications in telemetry
        nsga_specs = [("iris", 3, 240, 1), ("glass", 3, 120, 1), ("diabetes", 3, 48, 1)]
        cross_repeats = 2
        mortgage_cross_rows = 1000

    master_summary: Dict[str, Any] = {
        "mode": args.mode,
        "hardware": profiler.profile.__dict__,
        "normality_audit": {},
        "paper_datasets": {},
        "nsga2_experiments": {},
        "cross_model_benchmarks": {},
        "carbon_payback": {},
        "hotswap_bundles": {},
    }

    # 1. Exact 30-repeat Normality Audit (Paper Fig 1)
    for ds_key in ["iris", "diabetes"]:
        raw_p = RESULTS_DIR / f"normality_30rep_{ds_key}.csv"
        sum_p = RESULTS_DIR / f"normality_summary_{ds_key}.csv"
        if args.resume and raw_p.exists() and sum_p.exists():
            print(f"\n[Step 1/5] Loaded cached 30-repeat Normality Audit for {ds_key}", flush=True)
            raw_30 = pd.read_csv(raw_p)
            sum_df = pd.read_csv(sum_p)
        else:
            print(f"\n[Step 1/5] Running 30-repeat Normality Audit on {ds_key}...", flush=True)
            audit = run_normality_audit_30_repeats(ds_key, profiler, n_repeats=30, inference_repeats=inf_repeats)
            raw_30, sum_df = audit["raw_30rep"], audit["summary"]
        save_fig1_normality_histogram(raw_30, sum_df, ds_key)
        master_summary["normality_audit"][ds_key] = sum_df.to_dict(orient="records")

    # 2. Experiment 1 (Grid Search + Interleaving) & Experiment 2 (RQ2-4 Driver Analysis)
    paper_datasets = ["iris", "glass", "ionosphere", "diabetes", "mortgage"]
    exp1_cache: Dict[str, Dict[str, Any]] = {}
    for ds_key in paper_datasets:
        max_cfg, n_rep, do_cv, m_rows = ds_grid_cfg[ds_key]
        raw_csv = RESULTS_DIR / f"exp1_raw_{ds_key}.csv"
        agg_csv = RESULTS_DIR / f"exp1_agg_{ds_key}.csv"
        use_cached = False
        if args.resume and raw_csv.exists() and agg_csv.exists():
            tmp_agg = pd.read_csv(agg_csv)
            # Only reuse if hyperparams vary properly (alpha & max_iter non-constant)
            if tmp_agg["alpha"].nunique() > 1 and tmp_agg["max_iter"].nunique() > 1:
                use_cached = True

        if use_cached:
            print(f"\n[Step 2/5] Loaded cached Exp 1 for {ds_key} & recomputing Pareto/Knee/Exp 2...", flush=True)
            ds = load_dataset(ds_key, mortgage_max_rows=m_rows)
            raw_df = pd.read_csv(raw_csv)
            agg_df = pd.read_csv(agg_csv)
            agg_df = add_mlp_complexity_columns(ds, agg_df)
            pareto_train = extract_pareto_front(agg_df, acc_col="test_accuracy", energy_col="train_energy_j")
            pareto_inf = extract_pareto_front(agg_df, acc_col="test_accuracy", energy_col="inf_energy_j")
            agg_df, pareto_train, pareto_inf = compute_pareto_cv_if_missing(ds, agg_df, pareto_train, pareto_inf)
            knee_train = find_knee_point(pareto_train, acc_col="test_accuracy", energy_col="train_energy_j", co2_col="train_co2_g")
            knee_inf = find_knee_point(pareto_inf, acc_col="test_accuracy", energy_col="inf_energy_j", co2_col="inf_co2_g")
            agg_df.to_csv(agg_csv, index=False)
            pareto_train.to_csv(RESULTS_DIR / f"exp1_pareto_train_{ds_key}.csv", index=False)
            pareto_inf.to_csv(RESULTS_DIR / f"exp1_pareto_inf_{ds_key}.csv", index=False)

            med_t = float(agg_df["train_cpu_time_s"].median())
            sh = agg_df[agg_df["train_cpu_time_s"] <= med_t]
            lo = agg_df[agg_df["train_cpu_time_s"] > med_t]
            s_slope = float(np.polyfit(sh["train_cpu_time_s"], sh["train_energy_j"], 1)[0]) if len(sh) >= 2 else 0.0
            l_slope = float(np.polyfit(lo["train_cpu_time_s"], lo["train_energy_j"], 1)[0]) if len(lo) >= 2 else 0.0
            cc_path = RESULTS_DIR / "codecarbon_logs" / f"emissions_exp1_grid_{ds_key}.csv"
            cc_info = {"cached": True}
            if cc_path.exists():
                cc_df = pd.read_csv(cc_path)
                if not cc_df.empty:
                    cc_info = {
                        "duration_s": float(cc_df["duration"].iloc[-1]),
                        "codecarbon_energy_kwh": float(cc_df["energy_consumed"].iloc[-1]),
                        "codecarbon_co2_grams": float(cc_df["emissions"].iloc[-1]) * 1000.0,
                        "cpu_tracking_mode": "CodeCarbon OfflineEmissionsTracker",
                    }
            res1 = {
                "dataset": ds,
                "raw_df": raw_df,
                "agg_df": agg_df,
                "pareto_train": pareto_train,
                "pareto_inf": pareto_inf,
                "knee_train": knee_train,
                "knee_inf": knee_inf,
                "short_run_slope_j_per_s": s_slope,
                "long_run_slope_j_per_s": l_slope,
                "codecarbon_session": cc_info,
                "n_configs": int(len(agg_df)),
                "n_repeats": int(raw_df["repeat"].nunique()),
            }
        else:
            print(f"\n[Step 2/5] Exp 1 & 2 on {ds_key} (max_cfg={max_cfg}, repeats={n_rep}, cv={do_cv})...", flush=True)
            res1 = run_experiment_1_grid_search(
                dataset_key=ds_key,
                profiler=profiler,
                n_repeats=n_rep,
                inference_repeats=inf_repeats,
                compute_cv=do_cv,
                max_configs=max_cfg,
                mortgage_max_rows=m_rows,
            )
        exp1_cache[ds_key] = res1

        res2 = run_experiment_2_parameter_analysis(
            agg_df=res1["agg_df"],
            pareto_train=res1["pareto_train"],
            pareto_inf=res1["pareto_inf"],
            dataset_key=ds_key,
        )

        # Figures 2-8
        save_fig2_3_time_vs_energy(res1["agg_df"], ds_key)
        save_fig4_grid_spread(res1["agg_df"], ds_key)
        save_fig5_6_pareto_knee(res1["pareto_train"], res1["pareto_inf"], res1["knee_train"], res1["knee_inf"], ds_key)
        save_fig7_hyperparam_matrix(res1["pareto_train"], ds_key, stage="train")
        save_fig7_hyperparam_matrix(res1["pareto_inf"], ds_key, stage="inf")
        save_fig8_train_vs_inf(res1["agg_df"], ds_key)

        kt = res1["knee_train"]
        ki = res1["knee_inf"]
        print(
            f"  -> {ds_key}: configs={res1['n_configs']} | Train Knee: -{kt.energy_saved_pct:.1f}% energy (-{kt.accuracy_drop_pct_points:.2f}% acc) | "
            f"Inf Knee: -{ki.energy_saved_pct:.1f}% energy (-{ki.accuracy_drop_pct_points:.2f}% acc)",
            flush=True,
        )

        master_summary["paper_datasets"][ds_key] = {
            "source_note": res1["dataset"].source_note,
            "n_train": int(len(res1["dataset"].X_train)),
            "n_test": int(len(res1["dataset"].X_test)),
            "n_features": int(res1["dataset"].encoded_features),
            "n_configs": int(res1["n_configs"]),
            "n_repeats": int(res1["n_repeats"]),
            "knee_train": _knee_to_dict(kt),
            "knee_inf": _knee_to_dict(ki),
            "short_run_slope_j_per_s": round(res1["short_run_slope_j_per_s"], 4),
            "long_run_slope_j_per_s": round(res1["long_run_slope_j_per_s"], 4),
            "codecarbon_session": res1["codecarbon_session"],
            "rq2_spearman": res2["spearman_df"].to_dict(orient="records"),
            "rq3_categorical": res2["categorical_df"].to_dict(orient="records"),
            "rq4_summary": res2["rq4_summary"],
        }
        _save_summary(master_summary, t_start)

    # 3. Experiment 3: NSGA-II via jMetalPy (RQ5) + Carbon Payback
    for ds_key, nsga_runs, nsga_evals, nsga_repeats in nsga_specs:
        hist_p = RESULTS_DIR / f"exp3_nsga2_history_{ds_key}.csv"
        att_p = RESULTS_DIR / f"exp3_attainment_{ds_key}.csv"
        if args.resume and hist_p.exists() and att_p.exists():
            print(f"\n[Step 3/5] Loaded cached NSGA-II results for {ds_key}", flush=True)
            hist_df = pd.read_csv(hist_p)
            att_df = pd.read_csv(att_p)
            grid_pf = exp1_cache[ds_key]["pareto_train"]
            ref_e = max(float(hist_df["train_energy_j"].max()), float(exp1_cache[ds_key]["agg_df"]["train_energy_j"].max())) * 1.1
            hv_grid = compute_hypervolume_2d(grid_pf, acc_col="test_accuracy", energy_col="train_energy_j", ref_energy=ref_e)
            hv_runs = [
                compute_hypervolume_2d(
                    extract_pareto_front(grp, acc_col="test_accuracy", energy_col="train_energy_j"),
                    acc_col="test_accuracy",
                    energy_col="train_energy_j",
                    ref_energy=ref_e,
                )
                for _, grp in hist_df.groupby("run_id")
            ]
            cc_p = RESULTS_DIR / "codecarbon_logs" / f"emissions_exp3_nsga2_{ds_key}.csv"
            cc_info = {"cached": True}
            if cc_p.exists():
                cc_df = pd.read_csv(cc_p)
                if not cc_df.empty:
                    cc_info = {
                        "duration_s": float(cc_df["duration"].iloc[-1]),
                        "codecarbon_energy_kwh": float(cc_df["energy_consumed"].iloc[-1]),
                        "codecarbon_co2_grams": float(cc_df["emissions"].iloc[-1]) * 1000.0,
                        "cpu_tracking_mode": "CodeCarbon OfflineEmissionsTracker",
                    }
            res3 = {
                "history_df": hist_df,
                "attainment_df": att_df,
                "grid_pareto": grid_pf,
                "hv_grid": hv_grid,
                "hv_nsga2_median": float(np.median(hv_runs)) if hv_runs else 0.0,
                "hv_nsga2_best": float(np.max(hv_runs)) if hv_runs else 0.0,
                "hv_ratio_median": (float(np.median(hv_runs)) / hv_grid) if (hv_grid > 0 and hv_runs) else 0.0,
                "hv_ratio_best": (float(np.max(hv_runs)) / hv_grid) if (hv_grid > 0 and hv_runs) else 0.0,
                "codecarbon_session": cc_info,
            }
            nsga_runs_actual = int(hist_df["run_id"].nunique())
            nsga_evals_actual = int(hist_df.groupby("run_id").size().max())
        else:
            print(f"\n[Step 3/5] Exp 3 jMetalPy NSGA-II on {ds_key} ({nsga_runs} runs x {nsga_evals} evals)...", flush=True)
            res3 = run_experiment_3_nsga2(
                dataset_key=ds_key,
                profiler=profiler,
                n_runs=nsga_runs,
                max_evaluations=nsga_evals,
                repeats_per_eval=nsga_repeats,
                grid_agg_df=exp1_cache[ds_key]["agg_df"],
            )
            nsga_runs_actual = nsga_runs
            nsga_evals_actual = nsga_evals

        save_fig9_nsga2_attainment(res3["grid_pareto"], res3["attainment_df"], ds_key)

        master_summary["nsga2_experiments"][ds_key] = {
            "n_runs": nsga_runs_actual,
            "max_evaluations": nsga_evals_actual,
            "hv_grid": res3["hv_grid"],
            "hv_nsga2_median": res3["hv_nsga2_median"],
            "hv_nsga2_best": res3["hv_nsga2_best"],
            "hv_ratio_median": round(res3["hv_ratio_median"], 4),
            "hv_ratio_best": round(res3["hv_ratio_best"], 4),
            "codecarbon_session": res3["codecarbon_session"],
        }

        # Second-order Carbon Payback analysis
        grid_e_j = float(exp1_cache[ds_key]["raw_df"]["train_energy_j"].sum())
        nsga_e_j = float(res3["history_df"][res3["history_df"]["run_id"] == 0]["train_energy_j"].sum())
        ki = exp1_cache[ds_key]["knee_inf"]
        if ki is not None:
            payback = compute_carbon_payback(
                grid_search_energy_j=grid_e_j,
                nsga2_search_energy_j=nsga_e_j,
                baseline_inf_energy_1k_j=float(ki.max_accuracy_row["inf_energy_j"]),
                green_inf_energy_1k_j=float(ki.knee_row["inf_energy_j"]),
                n_test_samples=len(exp1_cache[ds_key]["dataset"].X_test),
                carbon_intensity_kg_per_kwh=profiler.profile.carbon_intensity_kg_per_kwh,
            )
            save_fig11_carbon_payback(payback, ds_key)
            master_summary["carbon_payback"][ds_key] = payback
        _save_summary(master_summary, t_start)

    # 4. Cross-Model Benchmark (Proposal Topic 1 + Paper Future Work) + Hot-Swap Surface
    cross_datasets = ["diabetes", "ionosphere", "mortgage", "telco_churn", "titanic"]
    for ds_key in cross_datasets:
        cm_p = RESULTS_DIR / f"cross_model_{ds_key}.csv"
        man_p = MODELS_DIR / f"{ds_key}_pareto_manifest.json"
        if args.resume and cm_p.exists() and man_p.exists():
            print(f"\n[Step 4/5] Loaded cached Cross-Model Benchmark for {ds_key}", flush=True)
            ds_cm = load_dataset(ds_key, mortgage_max_rows=(mortgage_cross_rows if ds_key == "mortgage" else None))
            cm_df = pd.read_csv(cm_p)
            pf_tr = extract_pareto_front(cm_df, acc_col="test_accuracy", energy_col="train_energy_j")
            pf_in = extract_pareto_front(cm_df, acc_col="test_accuracy", energy_col="inf_energy_j")
            kt_cm = find_knee_point(pf_tr, acc_col="test_accuracy", energy_col="train_energy_j", co2_col="train_co2_g")
            ki_cm = find_knee_point(pf_in, acc_col="test_accuracy", energy_col="inf_energy_j", co2_col="inf_co2_g")
            bundle = {"bundle_path": str(MODELS_DIR / f"{ds_key}_pareto_surface.joblib"), "manifest": json.loads(man_p.read_text(encoding="utf-8"))}
            cm = {"dataset": ds_cm, "agg_df": cm_df, "pareto_train": pf_tr, "knee_train": kt_cm, "knee_inf": ki_cm, "codecarbon_session": {"cached": True}}
        else:
            print(f"\n[Step 4/5] Cross-Model Benchmark on {ds_key}...", flush=True)
            cm = run_cross_model_benchmark(
                dataset_key=ds_key,
                profiler=profiler,
                n_repeats=cross_repeats,
                inference_repeats=inf_repeats,
                include_sweeps=True,
                mortgage_max_rows=(mortgage_cross_rows if ds_key == "mortgage" else None),
            )
            save_fig10_cross_model_tradeoff(cm["agg_df"], cm["pareto_train"], ds_key)
            bundle = export_pareto_hotswap_surface(
                dataset_key=ds_key,
                cross_df=cm["agg_df"],
                profiler=profiler,
                mortgage_max_rows=(mortgage_cross_rows if ds_key == "mortgage" else None),
            )
        master_summary["cross_model_benchmarks"][ds_key] = {
            "source_note": cm["dataset"].source_note,
            "knee_train": _knee_to_dict(cm["knee_train"]),
            "knee_inf": _knee_to_dict(cm["knee_inf"]),
            "codecarbon_session": cm["codecarbon_session"],
            "n_models_evaluated": int(len(cm["agg_df"])),
        }
        master_summary["hotswap_bundles"][ds_key] = bundle
        _save_summary(master_summary, t_start)

    elapsed = time.perf_counter() - t_start
    _save_summary(master_summary, t_start)
    print(f"\n[Step 5/5] Pipeline complete in {elapsed:.1f}s! Summary saved to {RESULTS_DIR / 'master_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
