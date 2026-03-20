"""
Channel Analysis & Attribution
===============================
Post-modelling analysis: revenue decomposition, ROI estimation,
marginal returns, saturation diagnostics, and Black Friday attribution.
"""

import numpy as np
import pandas as pd
import arviz as az

from src.transforms import (
    MEDIA_CHANNELS,
    geometric_adstock,
    hill_saturation,
    fourier_features,
    DECAY_RATES,
    HILL_PARAMS,
)


def decompose_revenue(
    trace: az.InferenceData,
    df: pd.DataFrame,
    model_data: dict,
) -> pd.DataFrame:
    """
    Decompose weekly revenue into baseline, channel contributions,
    promo uplift, and seasonality.

    Returns a DataFrame with columns:
      week_start_date, actual_revenue, baseline, {channel}_contribution,
      promo_contribution, seasonality, total_predicted
    """
    n = model_data["n"]
    y_mean = model_data["y_mean"]
    spend_scalers = model_data["spend_scalers"]
    fourier = model_data["fourier"]
    discount = model_data["discount"]

    posterior = trace.posterior

    # Take posterior means
    intercept = float(posterior["intercept"].mean())

    result = pd.DataFrame({
        "week_start_date": df["week_start_date"].values,
        "actual_revenue": df["total_revenue"].values,
    })

    # Baseline
    result["baseline"] = intercept * 1_000

    # Channel contributions
    total_channel = np.zeros(n)
    for ch in MEDIA_CHANNELS:
        beta = float(posterior[f"beta_{ch}"].mean())
        decay = float(posterior[f"decay_{ch}"].mean())
        half_max_val = float(posterior[f"half_max_{ch}"].mean())
        slope_val = float(posterior[f"slope_{ch}"].mean()) + 1.0

        raw = df[ch].values / spend_scalers[ch]
        adstocked = geometric_adstock(raw, decay)
        saturated = adstocked ** slope_val / (half_max_val ** slope_val + adstocked ** slope_val)
        contrib = beta * saturated * spend_scalers[ch]  # back to £

        result[f"{ch}_contribution"] = contrib
        total_channel += contrib

    # Promo contribution
    gamma = float(posterior["gamma_promo"].mean())
    promo_contrib = gamma * discount * y_mean * 1_000
    result["promo_contribution"] = promo_contrib

    # Seasonality
    season_coefs = posterior["season_coefs"].mean(dim=["chain", "draw"]).values
    seasonality = fourier @ season_coefs * 1_000
    result["seasonality"] = seasonality

    # Total predicted
    result["total_predicted"] = (
        result["baseline"] + total_channel + promo_contrib + seasonality
    )

    return result


def compute_channel_roi(
    trace: az.InferenceData,
    df: pd.DataFrame,
    model_data: dict,
) -> pd.DataFrame:
    """
    Compute ROI per channel: total attributed revenue / total spend.

    Returns DataFrame with columns:
      channel, total_spend, total_contribution, roi_mean,
      roi_lower_95, roi_upper_95
    """
    n = model_data["n"]
    spend_scalers = model_data["spend_scalers"]
    posterior = trace.posterior

    records = []
    for ch in MEDIA_CHANNELS:
        total_spend = df[ch].sum()

        # Sample across posterior draws for uncertainty
        beta_samples = posterior[f"beta_{ch}"].values.flatten()
        decay_samples = posterior[f"decay_{ch}"].values.flatten()
        half_max_samples = posterior[f"half_max_{ch}"].values.flatten()
        slope_samples = posterior[f"slope_{ch}"].values.flatten() + 1.0

        raw = df[ch].values / spend_scalers[ch]

        # Use posterior mean for point estimate
        decay_mean = float(decay_samples.mean())
        beta_mean = float(beta_samples.mean())
        hm_mean = float(half_max_samples.mean())
        sl_mean = float(slope_samples.mean())

        adstocked = geometric_adstock(raw, decay_mean)
        saturated = adstocked ** sl_mean / (hm_mean ** sl_mean + adstocked ** sl_mean)
        total_contrib = beta_mean * saturated.sum() * spend_scalers[ch]

        roi_mean = total_contrib / total_spend if total_spend > 0 else 0

        # Credible interval via sampling
        n_samples = min(500, len(beta_samples))
        roi_samples = []
        for i in range(n_samples):
            d = decay_samples[i]
            b = beta_samples[i]
            hm = half_max_samples[i]
            sl = slope_samples[i]
            adst = geometric_adstock(raw, d)
            sat = adst ** sl / (hm ** sl + adst ** sl)
            c = b * sat.sum() * spend_scalers[ch]
            roi_samples.append(c / total_spend if total_spend > 0 else 0)

        roi_samples = np.array(roi_samples)

        records.append({
            "channel": ch.replace("_spend", ""),
            "total_spend": total_spend,
            "total_contribution": total_contrib,
            "roi_mean": roi_mean,
            "roi_lower_95": np.percentile(roi_samples, 2.5),
            "roi_upper_95": np.percentile(roi_samples, 97.5),
        })

    return pd.DataFrame(records).sort_values("roi_mean", ascending=False)


def compute_marginal_roi(
    trace: az.InferenceData,
    df: pd.DataFrame,
    model_data: dict,
    increment: float = 1_000,
) -> pd.DataFrame:
    """
    Compute marginal ROI: incremental revenue from £1K additional spend
    at current spending levels, per channel.
    """
    spend_scalers = model_data["spend_scalers"]
    posterior = trace.posterior

    records = []
    for ch in MEDIA_CHANNELS:
        beta = float(posterior[f"beta_{ch}"].mean())
        decay = float(posterior[f"decay_{ch}"].mean())
        half_max_val = float(posterior[f"half_max_{ch}"].mean())
        slope_val = float(posterior[f"slope_{ch}"].mean()) + 1.0

        # Current mean weekly spend
        current_mean = df[ch].mean()
        raw_norm = current_mean / spend_scalers[ch]
        raw_inc = (current_mean + increment) / spend_scalers[ch]

        # Response at current and incremented spend (single-week, no adstock for simplicity)
        resp_current = raw_norm ** slope_val / (half_max_val ** slope_val + raw_norm ** slope_val)
        resp_inc = raw_inc ** slope_val / (half_max_val ** slope_val + raw_inc ** slope_val)

        marginal_revenue = beta * (resp_inc - resp_current) * spend_scalers[ch]
        marginal_roi = marginal_revenue / increment

        records.append({
            "channel": ch.replace("_spend", ""),
            "current_weekly_spend": current_mean,
            "marginal_roi_per_1k": marginal_roi,
            "marginal_revenue_per_1k": marginal_revenue,
        })

    return pd.DataFrame(records).sort_values("marginal_roi_per_1k", ascending=False)


def compute_saturation_curves(
    trace: az.InferenceData,
    model_data: dict,
    max_spend_mult: float = 3.0,
    n_points: int = 100,
) -> dict[str, pd.DataFrame]:
    """
    Generate saturation response curves for each channel.

    Returns dict of {channel: DataFrame} with columns:
      spend, response, current_spend_marker
    """
    spend_scalers = model_data["spend_scalers"]
    posterior = trace.posterior
    curves = {}

    for ch in MEDIA_CHANNELS:
        beta = float(posterior[f"beta_{ch}"].mean())
        half_max_val = float(posterior[f"half_max_{ch}"].mean())
        slope_val = float(posterior[f"slope_{ch}"].mean()) + 1.0

        max_spend = spend_scalers[ch] * max_spend_mult
        spends = np.linspace(0, max_spend, n_points)
        spends_norm = spends / spend_scalers[ch]

        response = spends_norm ** slope_val / (
            half_max_val ** slope_val + spends_norm ** slope_val
        )
        revenue = beta * response * spend_scalers[ch]

        curves[ch] = pd.DataFrame({
            "spend": spends,
            "response": revenue,
        })

    return curves


def black_friday_attribution(
    decomposition: pd.DataFrame,
    df: pd.DataFrame,
) -> dict:
    """
    Attribute Black Friday revenue spike to marketing vs organic.

    Returns dict with:
      - bf_total_revenue: actual revenue during BF week(s)
      - bf_baseline: organic/baseline revenue component
      - bf_marketing_driven: sum of channel contributions
      - bf_promo_driven: promo uplift component
      - pct_incremental: % of BF revenue that was marketing + promo driven
      - pct_organic: % that was baseline/organic demand
    """
    bf_mask = df["week_start_date"].dt.isocalendar().week.astype(int) == 47
    bf_decomp = decomposition[bf_mask.values]

    if len(bf_decomp) == 0:
        return {"error": "No Black Friday weeks found"}

    channel_cols = [c for c in bf_decomp.columns if c.endswith("_contribution") and c != "promo_contribution"]

    total_revenue = bf_decomp["actual_revenue"].sum()
    baseline = bf_decomp["baseline"].sum()
    seasonality = bf_decomp["seasonality"].sum()
    marketing = sum(bf_decomp[c].sum() for c in channel_cols)
    promo = bf_decomp["promo_contribution"].sum()

    organic = baseline + seasonality
    incremental = marketing + promo

    return {
        "bf_total_revenue": total_revenue,
        "bf_baseline_organic": organic,
        "bf_marketing_driven": marketing,
        "bf_promo_driven": promo,
        "pct_marketing_incremental": round(marketing / total_revenue * 100, 1) if total_revenue > 0 else 0,
        "pct_promo_incremental": round(promo / total_revenue * 100, 1) if total_revenue > 0 else 0,
        "pct_organic": round(organic / total_revenue * 100, 1) if total_revenue > 0 else 0,
        "n_bf_weeks": len(bf_decomp),
    }


def generate_full_analysis(
    trace: az.InferenceData,
    df: pd.DataFrame,
    model_data: dict,
) -> dict:
    """
    Run all analysis steps and return a dict of results.
    """
    decomposition = decompose_revenue(trace, df, model_data)
    roi_table = compute_channel_roi(trace, df, model_data)
    marginal_roi = compute_marginal_roi(trace, df, model_data)
    saturation_curves = compute_saturation_curves(trace, model_data)
    bf_attribution = black_friday_attribution(decomposition, df)

    return {
        "decomposition": decomposition,
        "roi_table": roi_table,
        "marginal_roi": marginal_roi,
        "saturation_curves": saturation_curves,
        "bf_attribution": bf_attribution,
    }
