"""
Streamlit Dashboard — Marketing Mix Model
==========================================
Interactive dashboard for exploring MMM results:
  Tab 1: Revenue Attribution (stacked area)
  Tab 2: Channel ROI (bar chart with credible intervals)
  Tab 3: Saturation Curves (interactive plotly sliders)
  Tab 4: Budget Optimizer (input budget → optimal allocation)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.transforms import MEDIA_CHANNELS
from src.analysis import (
    decompose_revenue,
    compute_channel_roi,
    compute_marginal_roi,
    compute_saturation_curves,
    black_friday_attribution,
)
from src.optimizer import optimise_budget

CHANNEL_COLOURS = {
    "tv": "#1f77b4",
    "paid_search": "#ff7f0e",
    "paid_social": "#2ca02c",
    "influencer": "#d62728",
    "email": "#9467bd",
    "affiliate": "#8c564b",
    "ooh": "#e377c2",
    "baseline": "#7f7f7f",
    "promo": "#bcbd22",
    "seasonality": "#17becf",
}

st.set_page_config(
    page_title="Marketing Mix Model — E-Commerce",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def load_artefacts():
    """Load pre-computed model artefacts."""
    import arviz as az
    import pickle

    artefacts_dir = ROOT / "artefacts"

    trace = az.from_netcdf(str(artefacts_dir / "trace.nc"))

    with open(artefacts_dir / "model_data.pkl", "rb") as f:
        model_data = pickle.load(f)

    df = pd.read_csv(ROOT / "data" / "ecommerce_data.csv", parse_dates=["week_start_date"])

    return trace, model_data, df


def main():
    st.title("📊 Marketing Mix Model — E-Commerce DTC Brand")
    st.markdown(
        "Moving beyond last-click attribution to understand **true channel contribution** "
        "to weekly revenue using Bayesian Marketing Mix Modelling."
    )

    try:
        trace, model_data, df = load_artefacts()
    except FileNotFoundError:
        st.error(
            "Model artefacts not found. Run `python run_model.py` first to generate "
            "the dataset, fit the model, and save artefacts."
        )
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Revenue Attribution",
        "💰 Channel ROI",
        "📉 Saturation Curves",
        "🎯 Budget Optimizer",
    ])

    # ── Tab 1: Revenue Attribution ────────────────────────────────
    with tab1:
        st.header("Revenue Attribution Over Time")
        st.markdown(
            "Stacked area chart showing how each channel, promotions, and "
            "seasonality contribute to total weekly revenue."
        )

        decomp = decompose_revenue(trace, df, model_data)

        # Build stacked area data
        contrib_cols = (
            ["baseline", "seasonality", "promo_contribution"]
            + [f"{ch}_contribution" for ch in MEDIA_CHANNELS]
        )
        display_names = (
            ["Baseline (Organic)", "Seasonality", "Promotions"]
            + [ch.replace("_spend", "").replace("_", " ").title() for ch in MEDIA_CHANNELS]
        )

        fig = go.Figure()
        for col, name in zip(contrib_cols, display_names):
            values = decomp[col].clip(lower=0)
            key = col.replace("_contribution", "").replace("_spend", "")
            colour = CHANNEL_COLOURS.get(key, "#636363")
            fig.add_trace(go.Scatter(
                x=decomp["week_start_date"],
                y=values,
                name=name,
                mode="lines",
                stackgroup="one",
                line=dict(width=0),
                fillcolor=colour,
            ))

        # Overlay actual revenue
        fig.add_trace(go.Scatter(
            x=decomp["week_start_date"],
            y=decomp["actual_revenue"],
            name="Actual Revenue",
            mode="lines",
            line=dict(color="black", width=2, dash="dot"),
        ))

        fig.update_layout(
            yaxis_title="Revenue (£)",
            xaxis_title="Week",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        total_actual = decomp["actual_revenue"].sum()
        total_predicted = decomp["total_predicted"].sum()
        mape = np.mean(np.abs(
            (decomp["actual_revenue"] - decomp["total_predicted"]) / decomp["actual_revenue"]
        )) * 100

        col1.metric("Total Actual Revenue", f"£{total_actual:,.0f}")
        col2.metric("Total Predicted Revenue", f"£{total_predicted:,.0f}")
        col3.metric("MAPE", f"{mape:.1f}%")

    # ── Tab 2: Channel ROI ────────────────────────────────────────
    with tab2:
        st.header("Channel ROI Analysis")
        st.markdown(
            "Return on Investment per channel with 95% credible intervals from "
            "the Bayesian posterior. RAG status: 🟢 ROI > 2.0, 🟡 1.0–2.0, 🔴 < 1.0."
        )

        roi_df = compute_channel_roi(trace, df, model_data)

        # RAG status
        def rag(roi):
            if roi >= 2.0:
                return "🟢 Strong"
            elif roi >= 1.0:
                return "🟡 Moderate"
            return "🔴 Weak"

        roi_df["rag_status"] = roi_df["roi_mean"].apply(rag)

        # Bar chart with error bars
        fig_roi = go.Figure()
        fig_roi.add_trace(go.Bar(
            x=roi_df["channel"],
            y=roi_df["roi_mean"],
            error_y=dict(
                type="data",
                symmetric=False,
                array=(roi_df["roi_upper_95"] - roi_df["roi_mean"]).values,
                arrayminus=(roi_df["roi_mean"] - roi_df["roi_lower_95"]).values,
            ),
            marker_color=[
                "#2ca02c" if r >= 2.0 else "#ff7f0e" if r >= 1.0 else "#d62728"
                for r in roi_df["roi_mean"]
            ],
            text=[f"{r:.2f}" for r in roi_df["roi_mean"]],
            textposition="outside",
        ))
        fig_roi.add_hline(y=1.0, line_dash="dash", line_color="gray",
                          annotation_text="Break-even (ROI = 1.0)")
        fig_roi.update_layout(
            yaxis_title="ROI (Revenue / Spend)",
            xaxis_title="Channel",
            height=450,
        )
        st.plotly_chart(fig_roi, use_container_width=True)

        # Table
        st.subheader("ROI Summary Table")
        display_df = roi_df[["channel", "total_spend", "total_contribution",
                             "roi_mean", "roi_lower_95", "roi_upper_95", "rag_status"]].copy()
        display_df.columns = ["Channel", "Total Spend (£)", "Attributed Revenue (£)",
                              "ROI (Mean)", "ROI (2.5%)", "ROI (97.5%)", "Status"]
        for col in ["Total Spend (£)", "Attributed Revenue (£)"]:
            display_df[col] = display_df[col].apply(lambda x: f"£{x:,.0f}")
        for col in ["ROI (Mean)", "ROI (2.5%)", "ROI (97.5%)"]:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
        st.dataframe(display_df, hide_index=True, use_container_width=True)

        # Marginal ROI
        st.subheader("Marginal ROI — Incremental Revenue per £1K Extra Spend")
        marginal = compute_marginal_roi(trace, df, model_data)
        fig_marginal = px.bar(
            marginal, x="channel", y="marginal_roi_per_1k",
            text="marginal_roi_per_1k",
            color="marginal_roi_per_1k",
            color_continuous_scale="RdYlGn",
        )
        fig_marginal.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_marginal.update_layout(
            yaxis_title="Marginal ROI (per £1K additional spend)",
            height=400,
        )
        st.plotly_chart(fig_marginal, use_container_width=True)

        # Black Friday insight
        st.subheader("🛍️ Black Friday Attribution")
        decomp_bf = decompose_revenue(trace, df, model_data)
        bf = black_friday_attribution(decomp_bf, df)
        if "error" not in bf:
            bcol1, bcol2, bcol3 = st.columns(3)
            bcol1.metric("BF Total Revenue", f"£{bf['bf_total_revenue']:,.0f}")
            bcol2.metric("Marketing-Driven", f"{bf['pct_marketing_incremental']}%")
            bcol3.metric("Organic Demand", f"{bf['pct_organic']}%")

    # ── Tab 3: Saturation Curves ──────────────────────────────────
    with tab3:
        st.header("Saturation Curves — Diminishing Returns")
        st.markdown(
            "See how each channel's response flattens as spend increases. "
            "The vertical dashed line shows your **current average weekly spend**."
        )

        sat_curves = compute_saturation_curves(trace, model_data)

        selected_channel = st.selectbox(
            "Select Channel",
            options=MEDIA_CHANNELS,
            format_func=lambda x: x.replace("_spend", "").replace("_", " ").title(),
        )

        curve_df = sat_curves[selected_channel]
        current_mean = df[selected_channel].mean()

        fig_sat = go.Figure()
        fig_sat.add_trace(go.Scatter(
            x=curve_df["spend"],
            y=curve_df["response"],
            mode="lines",
            name="Response Curve",
            line=dict(color="#1f77b4", width=3),
        ))
        fig_sat.add_vline(
            x=current_mean,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Current avg: £{current_mean:,.0f}",
        )
        fig_sat.update_layout(
            xaxis_title="Weekly Spend (£)",
            yaxis_title="Revenue Contribution (£)",
            height=450,
        )
        st.plotly_chart(fig_sat, use_container_width=True)

        # Spend slider
        st.subheader("What-if: Adjust Weekly Spend")
        max_val = int(curve_df["spend"].max())
        new_spend = st.slider(
            f"Hypothetical weekly {selected_channel.replace('_spend', '')} spend (£)",
            min_value=0,
            max_value=max_val,
            value=int(current_mean),
            step=500,
        )
        # Find response at new spend
        idx = np.argmin(np.abs(curve_df["spend"].values - new_spend))
        new_response = curve_df["response"].iloc[idx]
        idx_current = np.argmin(np.abs(curve_df["spend"].values - current_mean))
        current_response = curve_df["response"].iloc[idx_current]
        delta = new_response - current_response

        st.metric(
            f"Predicted Weekly Revenue from {selected_channel.replace('_spend', '').title()}",
            f"£{new_response:,.0f}",
            delta=f"£{delta:+,.0f} vs current",
        )

    # ── Tab 4: Budget Optimizer ───────────────────────────────────
    with tab4:
        st.header("🎯 Quarterly Budget Optimizer")
        st.markdown(
            "Set your total quarterly budget and target quarter, then see the "
            "optimal allocation that maximises predicted revenue."
        )

        ocol1, ocol2 = st.columns(2)
        with ocol1:
            total_budget = st.number_input(
                "Total Quarterly Budget (£)",
                min_value=100_000,
                max_value=5_000_000,
                value=500_000,
                step=50_000,
            )
        with ocol2:
            quarter = st.selectbox("Target Quarter", ["Q1", "Q2", "Q3", "Q4"], index=1)

        if st.button("🔍 Optimise Budget", type="primary"):
            with st.spinner("Running optimizer..."):
                result = optimise_budget(
                    trace, model_data, total_budget=total_budget, quarter=quarter
                )

            # Key metrics
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Current Expected Revenue", f"£{result['current_revenue']:,.0f}")
            mcol2.metric("Optimised Expected Revenue", f"£{result['expected_revenue']:,.0f}")
            mcol3.metric(
                "Revenue Uplift",
                f"{result['revenue_lift_pct']:+.1f}%",
                delta=f"£{result['expected_revenue'] - result['current_revenue']:+,.0f}",
            )

            # Allocation comparison
            st.subheader("Allocation Comparison")
            alloc = result["optimal_allocation"]

            fig_alloc = go.Figure()
            fig_alloc.add_trace(go.Bar(
                name="Current",
                x=alloc["channel"],
                y=alloc["current_allocation"],
                marker_color="#636363",
            ))
            fig_alloc.add_trace(go.Bar(
                name="Optimised",
                x=alloc["channel"],
                y=alloc["optimal_allocation"],
                marker_color="#2ca02c",
            ))
            fig_alloc.update_layout(
                barmode="group",
                yaxis_title="Quarterly Spend (£)",
                height=400,
            )
            st.plotly_chart(fig_alloc, use_container_width=True)

            # Pie chart of optimal allocation
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                st.subheader("Optimal Allocation")
                fig_pie = px.pie(
                    alloc, values="optimal_allocation", names="channel",
                    color="channel",
                    color_discrete_map={
                        ch.replace("_spend", ""): CHANNEL_COLOURS.get(ch.replace("_spend", ""), "#636363")
                        for ch in MEDIA_CHANNELS
                    },
                )
                fig_pie.update_traces(textinfo="percent+label")
                st.plotly_chart(fig_pie, use_container_width=True)

            with pcol2:
                st.subheader("Recommended Shifts")
                shifts_df = pd.DataFrame(result["recommended_shifts"])
                st.dataframe(shifts_df, hide_index=True, use_container_width=True)

            # Detailed table
            st.subheader("Detailed Allocation Table")
            display_alloc = alloc.copy()
            display_alloc["current_allocation"] = display_alloc["current_allocation"].apply(
                lambda x: f"£{x:,.0f}"
            )
            display_alloc["optimal_allocation"] = display_alloc["optimal_allocation"].apply(
                lambda x: f"£{x:,.0f}"
            )
            display_alloc["shift"] = display_alloc["shift"].apply(
                lambda x: f"£{x:+,.0f}"
            )
            st.dataframe(display_alloc, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
