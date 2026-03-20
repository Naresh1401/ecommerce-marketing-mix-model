"""
Media Transformations
=====================
Adstock (geometric decay) and saturation (Hill function) transforms
for marketing channel spend data. These capture the lagged and
diminishing-returns effects of advertising.
"""

import numpy as np
import pandas as pd

# ── Channel-specific decay rates ──────────────────────────────────
DECAY_RATES = {
    "tv_spend": 0.85,
    "paid_search_spend": 0.20,
    "paid_social_spend": 0.40,
    "influencer_spend": 0.60,
    "email_spend": 0.10,
    "affiliate_spend": 0.30,
    "ooh_spend": 0.50,
}

# ── Default Hill saturation parameters (half-max spend, slope) ────
HILL_PARAMS = {
    "tv_spend": {"half_max": 55_000, "slope": 1.8},
    "paid_search_spend": {"half_max": 35_000, "slope": 2.2},
    "paid_social_spend": {"half_max": 30_000, "slope": 2.0},
    "influencer_spend": {"half_max": 12_000, "slope": 1.5},
    "email_spend": {"half_max": 6_000, "slope": 2.5},
    "affiliate_spend": {"half_max": 10_000, "slope": 2.0},
    "ooh_spend": {"half_max": 18_000, "slope": 1.5},
}

MEDIA_CHANNELS = list(DECAY_RATES.keys())


def geometric_adstock(x: np.ndarray, decay: float) -> np.ndarray:
    """
    Apply geometric adstock transformation.

    Each week's effective spend is a blend of current spend and the
    decayed carryover from previous weeks — the 'advertising hangover'.

        adstocked[t] = x[t] + decay * adstocked[t-1]

    Parameters
    ----------
    x : array-like
        Raw weekly spend values.
    decay : float
        Decay rate in (0, 1). Higher = longer carryover.
        TV ≈ 0.85 (long memory), Search ≈ 0.2 (short memory).

    Returns
    -------
    np.ndarray
        Adstocked spend series.
    """
    x = np.asarray(x, dtype=np.float64)
    adstocked = np.zeros_like(x)
    adstocked[0] = x[0]
    for t in range(1, len(x)):
        adstocked[t] = x[t] + decay * adstocked[t - 1]
    return adstocked


def hill_saturation(x: np.ndarray, half_max: float, slope: float = 2.0) -> np.ndarray:
    """
    Hill (sigmoid) saturation function — models diminishing returns.

    At spend = half_max, the response is 50% of maximum. Beyond that,
    each additional pound generates less incremental return.

        response = x^slope / (half_max^slope + x^slope)

    Parameters
    ----------
    x : array-like
        Adstocked spend values.
    half_max : float
        Spend level at which response reaches 50% of its ceiling.
    slope : float
        Steepness of the saturation curve.

    Returns
    -------
    np.ndarray
        Saturated response in [0, 1].
    """
    x = np.asarray(x, dtype=np.float64)
    x = np.maximum(x, 0)
    return x ** slope / (half_max ** slope + x ** slope)


def promotion_multiplier(
    discount_depth_pct: np.ndarray,
    promo_lift_coef: float = 0.5,
) -> np.ndarray:
    """
    Compute promotional revenue lift multiplier.

        multiplier = 1 + promo_lift_coef × (discount_depth_pct / 100)

    A 20% discount with coef 0.5 yields a 10% revenue lift.

    Parameters
    ----------
    discount_depth_pct : array-like
        Weekly average discount percentage (0–50 typical).
    promo_lift_coef : float
        Elasticity of revenue to discount depth.

    Returns
    -------
    np.ndarray
        Multiplicative lift factor (≥ 1.0).
    """
    depth = np.asarray(discount_depth_pct, dtype=np.float64)
    return 1.0 + promo_lift_coef * depth / 100.0


def fourier_features(
    n: int,
    period: float = 52.18,
    n_harmonics: int = 2,
) -> np.ndarray:
    """
    Generate Fourier basis features for seasonality modelling.

    Parameters
    ----------
    n : int
        Number of time steps (weeks).
    period : float
        Period length in weeks (52.18 for annual seasonality).
    n_harmonics : int
        Number of sin/cos pairs to generate.

    Returns
    -------
    np.ndarray
        Shape (n, 2 * n_harmonics). Columns alternate sin, cos.
    """
    t = np.arange(n)
    features = []
    for h in range(1, n_harmonics + 1):
        features.append(np.sin(2 * np.pi * h * t / period))
        features.append(np.cos(2 * np.pi * h * t / period))
    return np.column_stack(features)


def transform_all_channels(
    df: pd.DataFrame,
    decay_rates: dict | None = None,
    hill_params: dict | None = None,
) -> pd.DataFrame:
    """
    Apply adstock and saturation to all media channels.

    Adds columns: {channel}_adstocked, {channel}_saturated for each channel.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain raw spend columns for each channel.
    decay_rates : dict, optional
        Override default decay rates.
    hill_params : dict, optional
        Override default Hill parameters.

    Returns
    -------
    pd.DataFrame
        Original dataframe with new transformed columns appended.
    """
    decay_rates = decay_rates or DECAY_RATES
    hill_params = hill_params or HILL_PARAMS

    df = df.copy()
    for channel in MEDIA_CHANNELS:
        raw = df[channel].values

        # Adstock
        decay = decay_rates[channel]
        adstocked = geometric_adstock(raw, decay)
        df[f"{channel}_adstocked"] = adstocked

        # Saturation
        hp = hill_params[channel]
        saturated = hill_saturation(adstocked, hp["half_max"], hp["slope"])
        df[f"{channel}_saturated"] = saturated

    return df
