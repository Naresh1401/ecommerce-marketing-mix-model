"""
Synthetic Retail Dataset Generator
===================================
Generates 156 weeks (3 years) of realistic e-commerce marketing data
with embedded seasonal patterns, Black Friday spikes, adstock effects,
and channel-specific ROI characteristics.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_WEEKS = 156  # 3 years


def generate_calendar(n_weeks: int = N_WEEKS) -> pd.DataFrame:
    """Create a weekly calendar starting from 2021-01-04 (Monday)."""
    dates = pd.date_range(start="2021-01-04", periods=n_weeks, freq="W-MON")
    df = pd.DataFrame({"week_start_date": dates})
    df["week_of_year"] = df["week_start_date"].dt.isocalendar().week.astype(int)
    df["month"] = df["week_start_date"].dt.month
    df["year"] = df["week_start_date"].dt.year
    return df


def add_season_index(df: pd.DataFrame) -> pd.DataFrame:
    """Season index: 1 (Jan low) to 5 (Nov-Dec peak)."""
    month_to_season = {
        1: 1, 2: 1.5, 3: 2, 4: 2.5, 5: 3, 6: 3,
        7: 3.5, 8: 3, 9: 3.5, 10: 4, 11: 4.5, 12: 5,
    }
    df["season_index"] = df["month"].map(month_to_season)
    return df


def add_promo_flags(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Flag key retail events: Black Friday (wk47), Boxing Day, Summer Sale."""
    df["promo_flag"] = 0
    df["discount_depth_pct"] = 0.0

    for _, row in df.iterrows():
        idx = row.name
        wk = row["week_of_year"]
        month = row["month"]

        # Black Friday week (week 47)
        if wk == 47:
            df.loc[idx, "promo_flag"] = 1
            df.loc[idx, "discount_depth_pct"] = rng.uniform(25, 40)
        # Boxing Day / post-Christmas (week 52 or week 1 of next year)
        elif wk == 52:
            df.loc[idx, "promo_flag"] = 1
            df.loc[idx, "discount_depth_pct"] = rng.uniform(20, 35)
        # Summer sale (mid-June to mid-July)
        elif month in (6, 7) and wk in range(24, 29):
            df.loc[idx, "promo_flag"] = 1
            df.loc[idx, "discount_depth_pct"] = rng.uniform(10, 25)
        # Easter / spring sale (April)
        elif month == 4 and wk in range(14, 17):
            df.loc[idx, "promo_flag"] = 1
            df.loc[idx, "discount_depth_pct"] = rng.uniform(10, 20)
        else:
            # Small random discounts on non-promo weeks (~15% chance)
            if rng.random() < 0.15:
                df.loc[idx, "discount_depth_pct"] = rng.uniform(3, 10)

    return df


def add_competitor_flags(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Competitors run promos roughly 20% of weeks, higher in Q4."""
    df["competitor_promo_flag"] = 0
    for idx, row in df.iterrows():
        threshold = 0.35 if row["month"] in (10, 11, 12) else 0.15
        if rng.random() < threshold:
            df.loc[idx, "competitor_promo_flag"] = 1
    return df


def add_sku_launches(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """New SKU launches: roughly 1 per quarter, more in Q3/Q4."""
    df["new_sku_launch_flag"] = 0
    for year in df["year"].unique():
        for quarter_months in [(2, 3), (5, 6), (8, 9), (10, 11)]:
            launch_month = rng.choice(quarter_months)
            candidates = df[(df["year"] == year) & (df["month"] == launch_month)]
            if len(candidates) > 0:
                launch_idx = rng.choice(candidates.index)
                df.loc[launch_idx, "new_sku_launch_flag"] = 1
    return df


def generate_media_spend(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Generate channel-specific weekly spend with seasonal scaling."""
    n = len(df)

    # Base weekly spend (£) — annual budget roughly £5.5M
    base_spend = {
        "tv_spend": 35_000,
        "paid_search_spend": 25_000,
        "paid_social_spend": 20_000,
        "influencer_spend": 8_000,
        "email_spend": 3_000,
        "affiliate_spend": 5_000,
        "ooh_spend": 10_000,
    }

    for channel, base in base_spend.items():
        seasonal_mult = 1.0 + 0.3 * (df["season_index"] - 3) / 2
        noise = rng.normal(1.0, 0.08, n)
        spend = base * seasonal_mult * noise

        # Black Friday weeks: 3× spend across all channels
        bf_mask = df["week_of_year"] == 47
        spend[bf_mask] *= 3.0

        # January pullback: -30% spend
        jan_mask = df["month"] == 1
        spend[jan_mask] *= 0.7

        # Email is relatively stable (owned channel)
        if channel == "email_spend":
            spend = base * rng.normal(1.0, 0.05, n)
            spend[bf_mask] *= 1.5  # Slight bump for Black Friday emails

        spend = np.maximum(spend, 0)
        df[channel] = np.round(spend, 2)

    return df


def generate_google_trends(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Brand search volume: correlated with TV spend + seasonality."""
    base_trend = 40 + 20 * (df["season_index"] - 1) / 4
    tv_effect = (df["tv_spend"] / df["tv_spend"].median()) * 10
    noise = rng.normal(0, 3, len(df))
    bf_boost = np.where(df["week_of_year"] == 47, 30, 0)

    df["google_trends_index"] = np.clip(
        base_trend + tv_effect + noise + bf_boost, 10, 100
    ).round(1)
    return df


def generate_revenue(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate revenue as a function of media spend, promos, and seasonality.
    Embeds realistic channel-specific effects and diminishing returns.
    """
    n = len(df)

    # --- Baseline organic revenue (brand equity + SEO) ---
    baseline = 180_000 * (1 + 0.15 * (df["season_index"] - 3) / 2)
    # Slight year-over-year growth (~8% annual)
    year_idx = (df["week_start_date"] - df["week_start_date"].iloc[0]).dt.days / 365.25
    baseline *= (1 + 0.08 * year_idx)

    # --- Channel contributions (with diminishing returns) ---
    def hill(x, half_max, slope=2):
        return x ** slope / (half_max ** slope + x ** slope)

    contributions = {}
    channel_params = {
        # (roi_coeff, half_max_spend, hill_slope)
        "tv_spend": (0.9, 60_000, 1.8),
        "paid_search_spend": (2.5, 40_000, 2.2),
        "paid_social_spend": (1.8, 35_000, 2.0),
        "influencer_spend": (1.5, 15_000, 1.5),
        "email_spend": (4.0, 8_000, 2.5),
        "affiliate_spend": (2.0, 12_000, 2.0),
        "ooh_spend": (0.7, 20_000, 1.5),
    }

    for channel, (roi, hm, slope) in channel_params.items():
        saturated = hill(df[channel].values, hm, slope)
        contributions[channel] = roi * saturated * df[channel].values

    total_media = sum(contributions.values())

    # --- Promo uplift ---
    promo_lift = 1.0 + 0.5 * df["discount_depth_pct"].values / 100

    # --- Competitor effect (slight negative) ---
    comp_effect = np.where(df["competitor_promo_flag"] == 1, 0.97, 1.0)

    # --- New SKU launch boost ---
    sku_boost = np.where(df["new_sku_launch_flag"] == 1, 1.05, 1.0)

    # --- Combine ---
    total_revenue = (baseline.values + total_media) * promo_lift * comp_effect * sku_boost

    # Black Friday multiplier on top (organic demand surge)
    bf_mask = df["week_of_year"] == 47
    total_revenue[bf_mask] *= 2.5  # On top of 3× spend -> ~4× total

    # January slump (organic demand drops)
    jan_mask = df["month"] == 1
    total_revenue[jan_mask] *= 0.75

    # Add noise
    total_revenue *= rng.normal(1.0, 0.03, n)
    total_revenue = np.maximum(total_revenue, 50_000)

    df["total_revenue"] = np.round(total_revenue, 2)

    # Split: ~85% online (DTC), ~15% store
    online_pct = rng.normal(0.85, 0.02, n)
    online_pct = np.clip(online_pct, 0.78, 0.92)
    df["online_revenue"] = np.round(df["total_revenue"] * online_pct, 2)
    df["store_revenue"] = np.round(df["total_revenue"] - df["online_revenue"], 2)

    # Units sold and AOV
    df["avg_order_value"] = np.round(rng.normal(65, 8, n), 2)
    df["avg_order_value"] = np.clip(df["avg_order_value"], 35, 120)
    # Promos lower AOV slightly
    df.loc[df["promo_flag"] == 1, "avg_order_value"] *= 0.9
    df["units_sold"] = np.round(df["total_revenue"] / df["avg_order_value"]).astype(int)

    # New vs returning customer revenue (~35% new, 65% returning)
    new_pct = rng.normal(0.35, 0.04, n)
    new_pct = np.clip(new_pct, 0.20, 0.50)
    # Black Friday / promos attract more new customers
    new_pct[bf_mask] += 0.1
    new_pct[df["promo_flag"].values == 1] += 0.05
    new_pct = np.clip(new_pct, 0.20, 0.55)
    df["new_customer_revenue"] = np.round(df["total_revenue"] * new_pct, 2)
    df["returning_customer_revenue"] = np.round(
        df["total_revenue"] - df["new_customer_revenue"], 2
    )

    return df


def generate_dataset(seed: int = SEED) -> pd.DataFrame:
    """Full pipeline to generate the synthetic retail dataset."""
    rng = np.random.default_rng(seed)

    df = generate_calendar()
    df = add_season_index(df)
    df = add_promo_flags(df, rng)
    df = add_competitor_flags(df, rng)
    df = add_sku_launches(df, rng)
    df = generate_media_spend(df, rng)
    df = generate_google_trends(df, rng)
    df = generate_revenue(df, rng)

    # Clean up helper columns
    cols_to_keep = [
        "week_start_date",
        "tv_spend", "paid_search_spend", "paid_social_spend",
        "influencer_spend", "email_spend", "affiliate_spend", "ooh_spend",
        "total_revenue", "online_revenue", "store_revenue",
        "units_sold", "avg_order_value",
        "new_customer_revenue", "returning_customer_revenue",
        "discount_depth_pct", "promo_flag", "competitor_promo_flag",
        "google_trends_index", "season_index", "new_sku_launch_flag",
    ]
    return df[cols_to_keep]


if __name__ == "__main__":
    output_path = Path(__file__).parent / "ecommerce_data.csv"
    df = generate_dataset()
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} weeks of data -> {output_path}")
    print(f"Date range: {df['week_start_date'].min()} to {df['week_start_date'].max()}")
    print(f"Revenue range: £{df['total_revenue'].min():,.0f} — £{df['total_revenue'].max():,.0f}")
    print(f"Mean weekly revenue: £{df['total_revenue'].mean():,.0f}")
