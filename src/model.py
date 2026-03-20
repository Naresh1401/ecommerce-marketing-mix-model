"""
Bayesian Marketing Mix Model
=============================
PyMC model with retail-informed priors, Fourier seasonality,
channel-specific Hill saturation parameters, and promo uplift.
"""

import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from src.transforms import (
    MEDIA_CHANNELS,
    fourier_features,
    geometric_adstock,
    hill_saturation,
)

# ── Prior configuration ───────────────────────────────────────────
CHANNEL_PRIORS = {
    "tv_spend": {"beta_sigma": 1.0},
    "paid_search_spend": {"beta_sigma": 2.0},
    "paid_social_spend": {"beta_sigma": 1.5},
    "influencer_spend": {"beta_sigma": 1.5},
    "email_spend": {"beta_sigma": 3.0},
    "affiliate_spend": {"beta_sigma": 1.5},
    "ooh_spend": {"beta_sigma": 0.8},
}


def build_model(
    df: pd.DataFrame,
    target_col: str = "total_revenue",
    n_harmonics: int = 2,
) -> tuple[pm.Model, dict]:
    """
    Construct the Bayesian MMM in PyMC.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain raw spend columns, discount_depth_pct, and target.
    target_col : str
        Column name for the response variable.
    n_harmonics : int
        Number of Fourier harmonics for annual seasonality.

    Returns
    -------
    model : pm.Model
    model_data : dict
        Pre-processed arrays for downstream analysis.
    """
    n = len(df)
    y = df[target_col].values

    # Scale revenue to thousands for numerical stability
    y_scaled = y / 1_000
    y_mean = y_scaled.mean()
    y_std = y_scaled.std()

    # Fourier seasonality features
    fourier = fourier_features(n, period=52.18, n_harmonics=n_harmonics)

    # Normalise spend to [0, 1] range for each channel
    spend_data = {}
    spend_scalers = {}
    for ch in MEDIA_CHANNELS:
        raw = df[ch].values.astype(np.float64)
        s_max = raw.max() if raw.max() > 0 else 1.0
        spend_data[ch] = raw / s_max
        spend_scalers[ch] = s_max

    # Discount depth normalised
    discount = df["discount_depth_pct"].values / 100.0

    coords = {"channel": MEDIA_CHANNELS, "fourier": [f"f_{i}" for i in range(fourier.shape[1])]}

    with pm.Model(coords=coords) as model:
        # ── Data containers ──
        spend_vars = {}
        for ch in MEDIA_CHANNELS:
            spend_vars[ch] = pm.Data(f"spend_{ch}", spend_data[ch])

        discount_data = pm.Data("discount_depth", discount)
        fourier_data = pm.Data("fourier_features", fourier)
        y_obs = pm.Data("y_observed", y_scaled)

        # ── Baseline (organic revenue) ──
        intercept = pm.Normal("intercept", mu=y_mean, sigma=y_std * 0.5)

        # ── Adstock decay priors ──
        decay = {}
        decay_priors = {
            "tv_spend": 0.85,
            "paid_search_spend": 0.20,
            "paid_social_spend": 0.40,
            "influencer_spend": 0.60,
            "email_spend": 0.10,
            "affiliate_spend": 0.30,
            "ooh_spend": 0.50,
        }
        for ch in MEDIA_CHANNELS:
            prior_mu = decay_priors[ch]
            decay[ch] = pm.Beta(
                f"decay_{ch}",
                alpha=prior_mu * 10,
                beta=(1 - prior_mu) * 10,
            )

        # ── Hill saturation parameters ──
        half_max = {}
        slope = {}
        for ch in MEDIA_CHANNELS:
            half_max[ch] = pm.HalfNormal(f"half_max_{ch}", sigma=0.5)
            slope[ch] = pm.HalfNormal(f"slope_{ch}", sigma=1.0) + 1.0  # slope ≥ 1

        # ── Channel betas (ROI coefficients) ──
        betas = {}
        for ch in MEDIA_CHANNELS:
            sigma = CHANNEL_PRIORS[ch]["beta_sigma"]
            betas[ch] = pm.HalfNormal(f"beta_{ch}", sigma=sigma)

        # ── Apply adstock + saturation and sum contributions ──
        channel_contrib = 0.0
        for ch in MEDIA_CHANNELS:
            # Scan-based adstock for gradient-friendly computation
            adstocked = _pytensor_adstock(spend_vars[ch], decay[ch], n)
            saturated = adstocked ** slope[ch] / (
                half_max[ch] ** slope[ch] + adstocked ** slope[ch]
            )
            channel_contrib = channel_contrib + betas[ch] * saturated * (spend_scalers[ch] / 1_000)

        # ── Promo uplift ──
        gamma = pm.HalfNormal("gamma_promo", sigma=0.5)
        promo_effect = gamma * discount_data * y_mean

        # ── Seasonality (Fourier) ──
        season_coefs = pm.Normal("season_coefs", mu=0, sigma=y_std * 0.3, shape=fourier.shape[1])
        seasonality = pm.math.dot(fourier_data, season_coefs)

        # ── Expected revenue ──
        mu = intercept + channel_contrib + promo_effect + seasonality

        # ── Likelihood ──
        sigma_obs = pm.HalfNormal("sigma", sigma=y_std * 0.5)
        pm.Normal("revenue", mu=mu, sigma=sigma_obs, observed=y_obs)

    model_data = {
        "y_scaled": y_scaled,
        "y_mean": y_mean,
        "y_std": y_std,
        "spend_scalers": spend_scalers,
        "spend_data": spend_data,
        "fourier": fourier,
        "discount": discount,
        "n": n,
    }

    return model, model_data


def _pytensor_adstock(x, decay, n):
    """Geometric adstock via pytensor scan for auto-differentiation."""
    import pytensor.tensor as pt
    from pytensor import scan

    result, _ = scan(
        fn=lambda current, previous, d: current + d * previous,
        sequences=[x],
        outputs_info=[pt.zeros(())],
        non_sequences=[decay],
        n_steps=n,
    )
    return result


def fit_model(
    model: pm.Model,
    draws: int = 2_000,
    tune: int = 500,
    chains: int = 2,
    target_accept: float = 0.9,
    random_seed: int = 42,
) -> az.InferenceData:
    """
    Run MCMC sampling and return the trace.

    Parameters
    ----------
    model : pm.Model
    draws : int
        Number of posterior draws per chain.
    tune : int
        Number of tuning steps.
    chains : int
        Number of MCMC chains.
    target_accept : float
        Target acceptance rate for NUTS.
    random_seed : int

    Returns
    -------
    az.InferenceData
        Posterior samples and posterior predictive checks.
    """
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
        )
        # Posterior predictive
        pm.sample_posterior_predictive(trace, extend_inferencedata=True)

    return trace


def save_trace(trace: az.InferenceData, path: str | Path) -> None:
    """Save the InferenceData object to NetCDF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    trace.to_netcdf(str(path))


def load_trace(path: str | Path) -> az.InferenceData:
    """Load a saved InferenceData object."""
    return az.from_netcdf(str(path))


def get_posterior_summary(trace: az.InferenceData) -> pd.DataFrame:
    """Return a summary table of key model parameters."""
    var_names = (
        [f"beta_{ch}" for ch in MEDIA_CHANNELS]
        + [f"decay_{ch}" for ch in MEDIA_CHANNELS]
        + ["intercept", "gamma_promo", "sigma"]
    )
    return az.summary(trace, var_names=var_names, hdi_prob=0.95)
