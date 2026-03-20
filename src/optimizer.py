"""
Budget Optimizer
================
Optimise quarterly marketing budget allocation across channels
using posterior response curves from the Bayesian MMM. Supports
seasonal adjustments (Q4 peak vs Q1 clearance) and minimum
spend constraints.
"""

import numpy as np
import pandas as pd
import arviz as az
from scipy.optimize import minimize

from src.transforms import MEDIA_CHANNELS, geometric_adstock


# ── Quarter seasonal multipliers ─────────────────────────────────
QUARTER_SEASONALITY = {
    "Q1": 0.75,   # January slump, clearance
    "Q2": 1.00,   # Baseline
    "Q3": 1.05,   # Back to school, slight lift
    "Q4": 1.40,   # Black Friday, Christmas peak
}

# 13 weeks per quarter
WEEKS_PER_QUARTER = 13


def _channel_response(
    weekly_spend: float,
    beta: float,
    decay: float,
    half_max: float,
    slope: float,
    spend_scaler: float,
    n_weeks: int = WEEKS_PER_QUARTER,
    seasonal_mult: float = 1.0,
) -> float:
    """
    Compute total quarterly revenue contribution from a single channel
    given a constant weekly spend level.

    Applies adstock accumulation over n_weeks and Hill saturation.
    """
    # Constant weekly spend vector
    raw = np.full(n_weeks, weekly_spend)
    raw_norm = raw / spend_scaler

    adstocked = geometric_adstock(raw_norm, decay)
    saturated = adstocked ** slope / (half_max ** slope + adstocked ** slope)
    total_response = beta * saturated.sum() * spend_scaler * seasonal_mult

    return total_response


def _total_revenue(
    quarterly_spends: np.ndarray,
    params: dict,
    seasonal_mult: float,
) -> float:
    """
    Compute total predicted revenue for a given budget allocation.

    Parameters
    ----------
    quarterly_spends : np.ndarray
        Total quarterly spend per channel (len = n_channels).
    params : dict
        Posterior parameter means per channel.
    seasonal_mult : float

    Returns
    -------
    float
        Total predicted revenue (negative for minimisation).
    """
    total = 0.0
    for i, ch in enumerate(MEDIA_CHANNELS):
        weekly_spend = quarterly_spends[i] / WEEKS_PER_QUARTER
        p = params[ch]
        total += _channel_response(
            weekly_spend=weekly_spend,
            beta=p["beta"],
            decay=p["decay"],
            half_max=p["half_max"],
            slope=p["slope"],
            spend_scaler=p["spend_scaler"],
            seasonal_mult=seasonal_mult,
        )
    return total


def extract_posterior_params(
    trace: az.InferenceData,
    model_data: dict,
) -> dict:
    """
    Extract posterior mean parameters for the optimizer.
    """
    posterior = trace.posterior
    params = {}
    for ch in MEDIA_CHANNELS:
        params[ch] = {
            "beta": float(posterior[f"beta_{ch}"].mean()),
            "decay": float(posterior[f"decay_{ch}"].mean()),
            "half_max": float(posterior[f"half_max_{ch}"].mean()),
            "slope": float(posterior[f"slope_{ch}"].mean()) + 1.0,
            "spend_scaler": model_data["spend_scalers"][ch],
        }
    return params


def optimise_budget(
    trace: az.InferenceData,
    model_data: dict,
    total_budget: float = 500_000,
    quarter: str = "Q2",
    min_channel_pct: float = 0.05,
    min_email_spend: float = 10_000,
) -> dict:
    """
    Find the optimal budget allocation to maximise total revenue.

    Parameters
    ----------
    trace : az.InferenceData
        Fitted model trace.
    model_data : dict
        Model data dict from build_model.
    total_budget : float
        Total quarterly budget in £.
    quarter : str
        One of Q1, Q2, Q3, Q4 — adjusts seasonal multiplier.
    min_channel_pct : float
        Minimum percentage of budget per channel (default 5%).
    min_email_spend : float
        Minimum email spend (owned channel constraint).

    Returns
    -------
    dict with keys:
        optimal_allocation: pd.DataFrame
        expected_revenue: float
        current_revenue: float
        revenue_lift_pct: float
        recommended_shifts: list of dicts
    """
    params = extract_posterior_params(trace, model_data)
    seasonal_mult = QUARTER_SEASONALITY.get(quarter, 1.0)
    n_channels = len(MEDIA_CHANNELS)

    # Current allocation (from recent data — last quarter)
    df_recent = pd.DataFrame(model_data["spend_data"])
    # Use last 13 weeks
    current_quarterly = {}
    for ch in MEDIA_CHANNELS:
        # spend_data is normalised, so multiply back
        scaler = model_data["spend_scalers"][ch]
        current_quarterly[ch] = df_recent[ch].iloc[-WEEKS_PER_QUARTER:].sum() * scaler

    current_total = sum(current_quarterly.values())

    # Scale current allocation to match total_budget
    current_scaled = {
        ch: v / current_total * total_budget for ch, v in current_quarterly.items()
    }

    # Compute revenue under current allocation
    current_spends = np.array([current_scaled[ch] for ch in MEDIA_CHANNELS])
    current_revenue = _total_revenue(current_spends, params, seasonal_mult)

    # ── Constraints & bounds ──
    min_spend = total_budget * min_channel_pct
    bounds = []
    for ch in MEDIA_CHANNELS:
        lower = max(min_spend, min_email_spend if ch == "email_spend" else 0)
        bounds.append((lower, total_budget * 0.5))  # No channel gets >50%

    # Budget equality constraint
    constraints = [
        {"type": "eq", "fun": lambda x: np.sum(x) - total_budget},
    ]

    # Initial guess: current allocation scaled
    x0 = current_spends.copy()

    # Objective: negative revenue (we minimise)
    def objective(x):
        return -_total_revenue(x, params, seasonal_mult)

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1_000, "ftol": 1e-10},
    )

    optimal_spends = result.x
    optimal_revenue = -result.fun

    # Build allocation table
    records = []
    shifts = []
    for i, ch in enumerate(MEDIA_CHANNELS):
        ch_name = ch.replace("_spend", "")
        current_val = current_spends[i]
        optimal_val = optimal_spends[i]
        diff = optimal_val - current_val

        records.append({
            "channel": ch_name,
            "current_allocation": round(current_val, 0),
            "optimal_allocation": round(optimal_val, 0),
            "current_pct": round(current_val / total_budget * 100, 1),
            "optimal_pct": round(optimal_val / total_budget * 100, 1),
            "shift": round(diff, 0),
        })

        direction = "+" if diff > 0 else ""
        shifts.append({
            "channel": ch_name,
            "recommendation": f"{direction}£{diff:,.0f}",
            "action": "Increase" if diff > 0 else "Decrease",
        })

    allocation_df = pd.DataFrame(records)
    revenue_lift_pct = (
        (optimal_revenue - current_revenue) / current_revenue * 100
        if current_revenue > 0 else 0
    )

    return {
        "optimal_allocation": allocation_df,
        "expected_revenue": optimal_revenue,
        "current_revenue": current_revenue,
        "revenue_lift_pct": round(revenue_lift_pct, 2),
        "recommended_shifts": shifts,
        "quarter": quarter,
        "total_budget": total_budget,
        "seasonal_multiplier": seasonal_mult,
    }


def compare_quarters(
    trace: az.InferenceData,
    model_data: dict,
    total_budget: float = 500_000,
) -> pd.DataFrame:
    """
    Compare optimal allocations across all four quarters.
    Returns a summary DataFrame.
    """
    all_results = []
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        result = optimise_budget(trace, model_data, total_budget, quarter=q)
        for _, row in result["optimal_allocation"].iterrows():
            all_results.append({
                "quarter": q,
                "channel": row["channel"],
                "optimal_allocation": row["optimal_allocation"],
                "optimal_pct": row["optimal_pct"],
            })

    return pd.DataFrame(all_results)
