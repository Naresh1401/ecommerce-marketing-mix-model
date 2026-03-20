"""
Run Model Pipeline
==================
End-to-end pipeline: generate data → fit Bayesian MMM → analyse → save artefacts.
"""

import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    # ── Step 1: Generate synthetic data ──
    print("=" * 60)
    print("STEP 1: Generating synthetic retail dataset (156 weeks)")
    print("=" * 60)

    from data.generate_data import generate_dataset

    df = generate_dataset()
    data_path = ROOT / "data" / "ecommerce_data.csv"
    df.to_csv(data_path, index=False)
    print(f"  Saved {len(df)} rows → {data_path}")
    print(f"  Revenue range: £{df['total_revenue'].min():,.0f} — £{df['total_revenue'].max():,.0f}")

    # ── Step 2: Transform media channels ──
    print("\n" + "=" * 60)
    print("STEP 2: Applying adstock & saturation transforms")
    print("=" * 60)

    from src.transforms import transform_all_channels

    df_transformed = transform_all_channels(df)
    print(f"  Added {len([c for c in df_transformed.columns if 'adstocked' in c or 'saturated' in c])} transformed columns")

    # ── Step 3: Build and fit Bayesian MMM ──
    print("\n" + "=" * 60)
    print("STEP 3: Building & fitting Bayesian MMM (PyMC)")
    print("  2 chains × 2000 draws + 500 tuning — this may take a while...")
    print("=" * 60)

    from src.model import build_model, fit_model, save_trace, get_posterior_summary

    model, model_data = build_model(df)
    trace = fit_model(model, draws=2_000, tune=500, chains=2)

    # Save artefacts
    artefacts_dir = ROOT / "artefacts"
    artefacts_dir.mkdir(exist_ok=True)

    trace_path = artefacts_dir / "trace.nc"
    save_trace(trace, trace_path)
    print(f"  Saved trace → {trace_path}")

    with open(artefacts_dir / "model_data.pkl", "wb") as f:
        pickle.dump(model_data, f)
    print(f"  Saved model_data → {artefacts_dir / 'model_data.pkl'}")

    # Posterior summary
    summary = get_posterior_summary(trace)
    print("\n  Posterior Summary (key params):")
    print(summary.to_string())

    # ── Step 4: Analysis ──
    print("\n" + "=" * 60)
    print("STEP 4: Running channel analysis & attribution")
    print("=" * 60)

    from src.analysis import generate_full_analysis

    results = generate_full_analysis(trace, df, model_data)

    # Save analysis results
    results["decomposition"].to_csv(artefacts_dir / "revenue_decomposition.csv", index=False)
    results["roi_table"].to_csv(artefacts_dir / "channel_roi.csv", index=False)
    results["marginal_roi"].to_csv(artefacts_dir / "marginal_roi.csv", index=False)

    print("\n  Channel ROI:")
    print(results["roi_table"][["channel", "roi_mean", "roi_lower_95", "roi_upper_95"]].to_string(index=False))

    print("\n  Black Friday Attribution:")
    bf = results["bf_attribution"]
    for k, v in bf.items():
        label = k.replace("_", " ").title()
        if isinstance(v, float):
            print(f"    {label}: £{v:,.0f}" if v > 100 else f"    {label}: {v}%")
        else:
            print(f"    {label}: {v}")

    # ── Step 5: Budget Optimisation ──
    print("\n" + "=" * 60)
    print("STEP 5: Optimising budget allocation (£500K/quarter)")
    print("=" * 60)

    from src.optimizer import optimise_budget

    opt_result = optimise_budget(trace, model_data, total_budget=500_000, quarter="Q2")
    opt_result["optimal_allocation"].to_csv(artefacts_dir / "optimal_allocation.csv", index=False)

    print(f"\n  Current expected revenue:   £{opt_result['current_revenue']:,.0f}")
    print(f"  Optimised expected revenue: £{opt_result['expected_revenue']:,.0f}")
    print(f"  Revenue uplift:             {opt_result['revenue_lift_pct']:+.1f}%")
    print("\n  Recommended shifts:")
    for shift in opt_result["recommended_shifts"]:
        print(f"    {shift['channel']:15s} → {shift['recommendation']}")

    # ── Done ──
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"  Artefacts saved to: {artefacts_dir}")
    print(f"  Launch dashboard:   streamlit run dashboard/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
