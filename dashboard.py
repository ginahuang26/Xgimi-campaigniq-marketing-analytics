"""
CampaignIQ — Streamlit marketing analytics dashboard.
Loads marketing_data.csv (same schema as analysis notebook).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Brand palette (consistent with notebook)
PALETTE = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"]


def _clean_currency(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"[RM,]", "", regex=True)
        .replace({"nan": np.nan})
        .astype(float)
    )


@st.cache_data(show_spinner=False)
def load_marketing_data(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Spend"] = _clean_currency(df["Acquisition_Cost"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Revenue"] = df["Spend"] * (1 + df["ROI"] / 100.0)
    df["Conversions"] = (df["Conversion_Rate"] * df["Clicks"]).clip(lower=0)
    df["AOV"] = np.where(df["Conversions"] > 0, df["Revenue"] / df["Conversions"], np.nan)
    return df


def channel_metrics(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("Channel_Used", as_index=False).agg(
        Spend=("Spend", "sum"),
        Revenue=("Revenue", "sum"),
        Conversions=("Conversions", "sum"),
        Clicks=("Clicks", "sum"),
    )
    g["ROI_pct"] = np.where(g["Spend"] > 0, (g["Revenue"] - g["Spend"]) / g["Spend"] * 100, np.nan)
    g["Conv_rate"] = np.where(g["Clicks"] > 0, g["Conversions"] / g["Clicks"], np.nan)
    return g


def budget_recommendation(d: pd.DataFrame) -> tuple[pd.DataFrame, float, float, float]:
    cm = channel_metrics(d)
    total_spend = float(cm["Spend"].sum())
    total_revenue = float(cm["Revenue"].sum())
    if total_spend <= 0 or cm.empty:
        return cm.assign(current_pct=np.nan, recommended_pct=np.nan), total_revenue, total_revenue, 0.0
    cm = cm.copy()
    cm["efficiency"] = np.where(cm["Spend"] > 0, cm["Revenue"] / cm["Spend"], 0.0)
    eff = cm["efficiency"].clip(lower=1e-9)
    raw = eff / eff.sum()
    cm["recommended_pct"] = raw
    cm["current_pct"] = cm["Spend"] / total_spend
    projected = float((total_spend * cm["recommended_pct"] * cm["efficiency"]).sum())
    lift_pct = (projected - total_revenue) / total_revenue * 100 if total_revenue > 0 else 0.0
    return cm, total_revenue, projected, lift_pct


def main() -> None:
    st.set_page_config(
        page_title="MarketPilot AI — Marketing Intelligence",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="📊",
    )

    st.title("🚀 MarketPilot AI — Marketing Intelligence Dashboard")
    st.caption("Malaysia Consumer Electronics | Marketing Analytics, Customer Insights & Budget Optimization")

    # ── File uploader ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Data")
        uploaded = st.file_uploader(
            "Upload your dataset CSV",
            type=["csv"],
            help="Upload marketing_data.csv",
        )

    if uploaded is None:
        st.info("👈 Upload your dataset CSV using the sidebar to get started.")
        st.stop()

    df = load_marketing_data(uploaded)

    # ── Filters (sidebar "slicers") ───────────────────────────────────────────
    with st.sidebar:
        st.header("Filters")
        min_d, max_d = df["Date"].min(), df["Date"].max()
        dr = st.date_input(
            "Date range",
            value=(min_d.date(), max_d.date()),
            min_value=min_d.date(),
            max_value=max_d.date(),
        )
        if isinstance(dr, tuple) and len(dr) == 2:
            d0, d1 = pd.Timestamp(dr[0]), pd.Timestamp(dr[1])
        elif hasattr(dr, "year"):
            d0 = d1 = pd.Timestamp(dr)
        else:
            d0, d1 = min_d, max_d

        ctype = st.multiselect(
            "Campaign type",
            options=sorted(df["Campaign_Type"].dropna().unique()),
            default=sorted(df["Campaign_Type"].dropna().unique()),
        )
        channels = st.multiselect(
            "Channels",
            options=sorted(df["Channel_Used"].dropna().unique()),
            default=sorted(df["Channel_Used"].dropna().unique()),
        )
        segments = st.multiselect(
            "Audience segment",
            options=sorted(df["Customer_Segment"].dropna().unique()),
            default=sorted(df["Customer_Segment"].dropna().unique()),
        )

        audiences = None
        if "Target_Audience" in df.columns:
            audiences = st.multiselect(
                "Target audience",
                options=sorted(df["Target_Audience"].dropna().unique()),
                default=sorted(df["Target_Audience"].dropna().unique()),
            )

    filt = df[
        (df["Date"] >= d0)
        & (df["Date"] <= d1)
        & (df["Campaign_Type"].isin(ctype))
        & (df["Channel_Used"].isin(channels))
        & (df["Customer_Segment"].isin(segments))
    ]
    if audiences is not None:
        filt = filt[filt["Target_Audience"].isin(audiences)]

    if filt.empty:
        st.warning("No rows match the selected filters.")
        st.stop()

    total_spend = float(filt["Spend"].sum())
    total_revenue = float(filt["Revenue"].sum())
    overall_roi = (total_revenue - total_spend) / total_spend * 100 if total_spend > 0 else 0.0
    total_conversions = float(filt["Conversions"].sum())
    total_clicks = float(filt["Clicks"].sum())
    overall_conv_rate = (total_conversions / total_clicks * 100) if total_clicks > 0 else 0.0
    overall_roas = total_revenue / total_spend if total_spend > 0 else 0.0
    overall_cpa = total_spend / total_conversions if total_conversions > 0 else 0.0
    cm = channel_metrics(filt)
    best_channel = (
        cm.sort_values("ROI_pct", ascending=False).iloc[0]["Channel_Used"]
        if not cm.empty else "—"
    )

    # ── KPI cards ──────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)

    k1.metric("Marketing Spend", f"RM {total_spend:,.0f}")
    k2.metric("Attributed Revenue", f"RM {total_revenue:,.0f}")
    k3.metric("ROAS", f"{overall_roas:.2f}x")
    k4.metric("Conversions", f"{total_conversions:,.0f}")
    k5.metric("CPA", f"RM {overall_cpa:,.0f}")
    k6.metric("Best Channel", best_channel)

    st.markdown("---")

    tab_overview, tab_channels, tab_products,tab_campaigns, tab_customers,tab_budget = st.tabs(
        ["Overview","Channels & Segments","Product Performance","Campaign Performance","Customers Intelligence","Budget Optimization"]
    )

    # ── Tab 1: Overview ──────────────────────────────────────────────────────
    with tab_overview:
        st.caption("How spend and revenue are trending, and where campaigns are concentrated.")

        m = filt.assign(month=filt["Date"].dt.to_period("M").dt.to_timestamp())
        monthly = m.groupby("month", as_index=False).agg(Spend=("Spend", "sum"), Revenue=("Revenue", "sum"))
        trend = go.Figure()
        trend.add_trace(go.Scatter(
            x=monthly["month"], y=monthly["Spend"], name="Spend",
            line=dict(color=PALETTE[0], width=2), yaxis="y1",
        ))
        trend.add_trace(go.Scatter(
            x=monthly["month"], y=monthly["Revenue"], name="Revenue",
            line=dict(color=PALETTE[1], width=2), yaxis="y2",
        ))
        trend.update_layout(
            title="Monthly spend vs revenue", height=400,
            yaxis=dict(title="Spend (RM)", side="left", showgrid=False),
            yaxis2=dict(title="Revenue (RM)", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(trend, use_container_width=True)

        scatter = px.scatter(
            filt, x="Spend", y="Revenue", size="Conversions", color="Channel_Used",
            hover_data=["Campaign_Type", "Customer_Segment"],
            color_discrete_sequence=PALETTE,
            title="Spend vs Revenue (bubble size = conversions)",
        )
        scatter.update_layout(height=420, legend_title_text="Channel")
        st.plotly_chart(scatter, use_container_width=True)

    # ── Tab 2: Channels & Segments ───────────────────────────────────────────
    with tab_channels:
        st.caption("Compare channels and audience segments to see where ROI is strongest.")
        st.subheader("Channel Performance Overview")

        channel_summary = (
        filt.groupby("Channel_Used", as_index=False)
        .agg(
            Spend=("Spend", "sum"),
            Revenue=("Revenue", "sum"),
            Conversions=("Conversions", "sum"),
            Clicks=("Clicks", "sum"),
        )
    )

        channel_summary["ROAS"] = (
        channel_summary["Revenue"] / channel_summary["Spend"]
    )

        channel_summary["CPA"] = (
        channel_summary["Spend"]
        / channel_summary["Conversions"].replace(0, float("nan"))
    )

        channel_summary["Conversion_Rate"] = (
        channel_summary["Conversions"]
        / channel_summary["Clicks"]
        * 100
    )
        best_roas_channel = channel_summary.loc[
        channel_summary["ROAS"].idxmax()
    ]

        best_cpa_channel = channel_summary.loc[
        channel_summary["CPA"].idxmin()
    ]

        best_conversion_channel = channel_summary.loc[
        channel_summary["Conversion_Rate"].idxmax()
    ]
        c1, c2, c3 = st.columns(3)

        c1.metric(
        "Highest ROAS",
        best_roas_channel["Channel_Used"],
        f'{best_roas_channel["ROAS"]:.2f}x'
    )

        c2.metric(
        "Lowest CPA",
        best_cpa_channel["Channel_Used"],
        f'RM {best_cpa_channel["CPA"]:,.0f}'
    )

        c3.metric(
        "Best Conversion Rate",
        best_conversion_channel["Channel_Used"],
        f'{best_conversion_channel["Conversion_Rate"]:.2f}%'
    )
    
        st.subheader("Channel Performance Table")

        channel_display = channel_summary.copy()

        channel_display["Spend"] = channel_display["Spend"].round(0)
        channel_display["Revenue"] = channel_display["Revenue"].round(0)
        channel_display["ROAS"] = channel_display["ROAS"].round(2)
        channel_display["CPA"] = channel_display["CPA"].round(0)
        channel_display["Conversion_Rate"] = (
            channel_display["Conversion_Rate"].round(2)
    )

        channel_display = channel_display.rename(
            columns={
                "Channel_Used": "Channel",
                "Spend": "Spend (RM)",
                "Revenue": "Revenue (RM)",
                "ROAS": "ROAS (x)",
                "CPA": "CPA (RM)",
                "Conversion_Rate": "Conversion Rate (%)"
        }
    )

        channel_display = channel_display[
        [
            "Channel",
            "Spend (RM)",
            "Revenue (RM)",
            "ROAS (x)",
            "CPA (RM)",
            "Conversion Rate (%)"
        ]
    ]

        st.dataframe(
            channel_display.sort_values(
                "ROAS (x)",
                ascending=False
        ),
            use_container_width=True,
            hide_index=True
    )
        st.subheader("ROAS by Marketing Channel")

        roas_chart_data = channel_summary.sort_values(
            "ROAS",
            ascending=False
    )

        fig_roas = px.bar(
            roas_chart_data,
            x="Channel_Used",
            y="ROAS",
            text="ROAS",
            labels={
            "Channel_Used": "Marketing Channel",
            "ROAS": "ROAS (x)"
        }
    )

        fig_roas.update_traces(
            texttemplate="%{text:.2f}x",
            textposition="outside"
    )

        fig_roas.update_layout(
            xaxis_title="Marketing Channel",
            yaxis_title="ROAS (x)",
            height=430
    )

        st.plotly_chart(
            fig_roas,
            use_container_width=True
    )
        st.subheader("Channel Efficiency Matrix")

        avg_spend = channel_summary["Spend"].mean()
        avg_revenue = channel_summary["Revenue"].mean()

        fig_matrix = px.scatter(
            channel_summary,
            x="Spend",
            y="Revenue",
            size="Conversions",
            text="Channel_Used",
            hover_name="Channel_Used",
            labels={
                "Spend": "Marketing Spend (RM)",
                "Revenue": "Attributed Revenue (RM)",
                "Conversions": "Conversions"
        }
    )

        fig_matrix.add_vline(
            x=avg_spend,
            line_dash="dash"
    )

        fig_matrix.add_hline(
            y=avg_revenue,
            line_dash="dash"
    )

        fig_matrix.update_traces(
            textposition="top center"
    )

        fig_matrix.update_layout(
            height=500,
            xaxis_title="Marketing Spend (RM)",
            yaxis_title="Attributed Revenue (RM)"
    )

        st.plotly_chart(
            fig_matrix,
            use_container_width=True
    )
        worst_roas_channel = channel_summary.loc[
            channel_summary["ROAS"].idxmin()
    ]

        highest_revenue_channel = channel_summary.loc[
            channel_summary["Revenue"].idxmax()
    ]

        st.subheader("Business Insights")

        st.info(
            f"""
            **Channel Performance Summary**

            • **{best_roas_channel["Channel_Used"]}** delivers the highest ROAS at
            **{best_roas_channel["ROAS"]:.2f}x**, indicating the strongest return on marketing investment.

            • **{best_cpa_channel["Channel_Used"]}** achieves the lowest CPA at
            **RM {best_cpa_channel["CPA"]:,.0f}**, suggesting stronger conversion cost efficiency.

            • **{highest_revenue_channel["Channel_Used"]}** contributes the highest attributed revenue,
            generating approximately **RM {highest_revenue_channel["Revenue"]:,.0f}**.
            • **{worst_roas_channel["Channel_Used"]}** currently records the lowest ROAS at
            **{worst_roas_channel["ROAS"]:.2f}x** and should be reviewed for targeting,
            creative, bidding, or budget allocation efficiency.
            """
    )
        sort_metric = st.radio(
            "Rank channels by", ["ROI %", "Revenue", "Conversion Rate"], horizontal=True,
        )
        metric_map = {"ROI %": "ROI_pct", "Revenue": "Revenue", "Conversion Rate": "Conv_rate"}
        cm_sorted = cm.sort_values(metric_map[sort_metric], ascending=True)
        roi_bar = px.bar(
            cm_sorted, x=metric_map[sort_metric], y="Channel_Used", orientation="h",
            color=metric_map[sort_metric], color_continuous_scale=PALETTE,
            title=f"Channel ranking by {sort_metric}",
            labels={metric_map[sort_metric]: sort_metric, "Channel_Used": "Channel"},
        )
        roi_bar.update_layout(height=420, showlegend=False)
        st.plotly_chart(roi_bar, use_container_width=True)

        heat = (
            filt.groupby(["Customer_Segment", "Channel_Used"], as_index=False)
            .agg(roi=("ROI", "mean"))
            .pivot(index="Customer_Segment", columns="Channel_Used", values="roi")
        )
        heatmap = px.imshow(
            heat, aspect="auto", color_continuous_scale=PALETTE,
            title="Average campaign ROI by segment × channel",
            labels=dict(x="Channel", y="Segment", color="Avg ROI"),
        )
        heatmap.update_layout(height=420)
        st.plotly_chart(heatmap, use_container_width=True)
    

    # ── Tab 3: Product Performance ───────────────────────────────────────────

    with tab_products:
        st.caption(
            "Compare product performance across revenue, ROAS, "
            "conversion efficiency and product positioning."
        )

        st.subheader("Product Performance Overview")
        product_summary = (
            filt.groupby(
                ["Product_Model", "Product_Category", "Product_Tier"],
                as_index=False
                )
            .agg(
                Spend=("Spend", "sum"),
                Revenue=("Revenue", "sum"),
                Conversions=("Conversions", "sum"),
                Clicks=("Clicks", "sum"),
                )
        )

        product_summary["ROAS"] = (
            product_summary["Revenue"] / product_summary["Spend"]
        )

        product_summary["CPA"] = (
            product_summary["Spend"]
            / product_summary["Conversions"].replace(0, float("nan"))
        )

        product_summary["Conversion_Rate"] = (
            product_summary["Conversions"]
            / product_summary["Clicks"]
            * 100
        )
        best_product_roas = product_summary.loc[
            product_summary["ROAS"].idxmax()
        ]

        best_product_revenue = product_summary.loc[
            product_summary["Revenue"].idxmax()
        ]

        best_product_conversion = product_summary.loc[
            product_summary["Conversion_Rate"].idxmax()
        ]

        lowest_product_cpa = product_summary.loc[
            product_summary["CPA"].idxmin()
        ]
        
        p1, p2, p3, p4 = st.columns(4)
        p1.metric(
            "Highest ROAS Product",
            best_product_roas["Product_Model"],
            f'{best_product_roas["ROAS"]:.2f}x'
        )

        p2.metric(
            "Revenue Leader",
            best_product_revenue["Product_Model"],
            f'RM {best_product_revenue["Revenue"]:,.0f}'
        )

        p3.metric(
            "Best Conversion Product",
            best_product_conversion["Product_Model"],
            f'{best_product_conversion["Conversion_Rate"]:.2f}%'
        )

        p4.metric(
            "Lowest CPA Product",
            lowest_product_cpa["Product_Model"],
            f'RM {lowest_product_cpa["CPA"]:,.0f}'
        )
    
        st.subheader("Revenue & ROAS by Product")
        product_chart = px.bar(
            product_summary.sort_values(
                "Revenue",
                ascending=False
            ),
            x="Product_Model",
            y="Revenue",
            text="Revenue",
            color="ROAS",
            labels={
                "Product_Model": "Product",
                "Revenue": "Attributed Revenue (RM)",
                "ROAS": "ROAS (x)"
            }
        )

        product_chart.update_traces(
            texttemplate="RM %{text:,.0f}",
            textposition="outside"
        )

        product_chart.update_layout(
            height=470,
            xaxis_title="Product",
            yaxis_title="Attributed Revenue (RM)"
        )

        st.plotly_chart(
            product_chart,
            use_container_width=True
        )
    
        st.subheader("Performance by Product Tier")
        tier_summary = (
            filt.groupby(
                "Product_Tier",
                as_index=False
            )
            .agg(
                Spend=("Spend", "sum"),
                Revenue=("Revenue", "sum"),
                Conversions=("Conversions", "sum"),
            )
        )

        tier_summary["ROAS"] = (
            tier_summary["Revenue"] / tier_summary["Spend"]
        )

        tier_summary["CPA"] = (
            tier_summary["Spend"]
            / tier_summary["Conversions"].replace(0, float("nan"))
        )

        tier_chart = px.bar(
            tier_summary,
            x="Product_Tier",
            y="Revenue",
            color="ROAS",
            text="Revenue",
            labels={
                "Product_Tier": "Product Tier",
                "Revenue": "Attributed Revenue (RM)",
                "ROAS": "ROAS (x)"
            }
        )

        tier_chart.update_traces(
            texttemplate="RM %{text:,.0f}",
            textposition="outside"
        )

        tier_chart.update_layout(
            height=420
        )

        st.plotly_chart(
            tier_chart,
            use_container_width=True
        )
    
        st.subheader("Product Performance Table")

        product_display = product_summary.copy()

        product_display["Spend"] = product_display["Spend"].round(0)
        product_display["Revenue"] = product_display["Revenue"].round(0)
        product_display["ROAS"] = product_display["ROAS"].round(2)
        product_display["CPA"] = product_display["CPA"].round(0)
        product_display["Conversion_Rate"] = (
            product_display["Conversion_Rate"].round(2)
        )

        product_display = product_display.rename(
            columns={
                "Product_Model": "Product",
                "Product_Category": "Category",
                "Product_Tier": "Tier",
                "Spend": "Spend (RM)",
                "Revenue": "Revenue (RM)",
                "ROAS": "ROAS (x)",
                "CPA": "CPA (RM)",
                "Conversion_Rate": "Conversion Rate (%)"
            }
        )

        st.dataframe(
            product_display.sort_values(
                "Revenue (RM)",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

        worst_product_roas = product_summary.loc[
            product_summary["ROAS"].idxmin()
        ]

        st.subheader("Product Insights")

        st.info(
            f"""
            **Product Performance Summary**

            • **{best_product_revenue["Product_Model"]}** is the revenue leader,
            generating approximately **RM {best_product_revenue["Revenue"]:,.0f}**
            in attributed revenue.

            • **{best_product_roas["Product_Model"]}** delivers the strongest marketing efficiency
            with a ROAS of **{best_product_roas["ROAS"]:.2f}x**.

            • **{best_product_conversion["Product_Model"]}** achieves the highest conversion rate
            at **{best_product_conversion["Conversion_Rate"]:.2f}%**.

            • **{worst_product_roas["Product_Model"]}** has the lowest ROAS at
            **{worst_product_roas["ROAS"]:.2f}x**, suggesting that its targeting,
            pricing, promotional strategy or channel mix may require optimization.
            """
        )

    # ── Tab 5: Campaign Performance ─────────────────────────────────────────
    with tab_campaigns:
        st.caption(
            "Evaluate seasonal and promotional campaigns across revenue, "
            "ROAS, conversion efficiency and marketing spend."
        )

        st.subheader("Campaign Performance Overview")
        campaign_data = filt.copy()

        campaign_data["Campaign_Theme"] = (
            campaign_data["Campaign_Name"]
            .str.split(" | ", regex=False)
            .str[0]
        )
        campaign_summary = (
            campaign_data.groupby(
                "Campaign_Theme",
                as_index=False
            )
            .agg(
                Spend=("Spend", "sum"),
                Revenue=("Revenue", "sum"),
                Conversions=("Conversions", "sum"),
                Clicks=("Clicks", "sum"),
            )
        )

        campaign_summary["ROAS"] = (
            campaign_summary["Revenue"]
            / campaign_summary["Spend"]
        )

        campaign_summary["CPA"] = (
            campaign_summary["Spend"]
            / campaign_summary["Conversions"].replace(0, float("nan"))
        )

        campaign_summary["Conversion_Rate"] = (
            campaign_summary["Conversions"]
            / campaign_summary["Clicks"]
            * 100
        )
        best_campaign_roas = campaign_summary.loc[
            campaign_summary["ROAS"].idxmax()
        ]

        best_campaign_revenue = campaign_summary.loc[
            campaign_summary["Revenue"].idxmax()
        ]

        best_campaign_conversion = campaign_summary.loc[
            campaign_summary["Conversion_Rate"].idxmax()
        ]

        lowest_campaign_cpa = campaign_summary.loc[
            campaign_summary["CPA"].idxmin()
        ]

        ca1, ca2, ca3, ca4 = st.columns(4)

        ca1.metric(
            "Highest ROAS Campaign",
            best_campaign_roas["Campaign_Theme"],
            f'{best_campaign_roas["ROAS"]:.2f}x'
        )

        ca2.metric(
            "Revenue Leader",
            best_campaign_revenue["Campaign_Theme"],
            f'RM {best_campaign_revenue["Revenue"]:,.0f}'
        )

        ca3.metric(
            "Best Conversion Campaign",
            best_campaign_conversion["Campaign_Theme"],
            f'{best_campaign_conversion["Conversion_Rate"]:.2f}%'
        )

        ca4.metric(
            "Lowest CPA Campaign",
            lowest_campaign_cpa["Campaign_Theme"],
            f'RM {lowest_campaign_cpa["CPA"]:,.0f}'
        )

        st.subheader("Revenue & ROAS by Campaign")

        campaign_chart = px.bar(
            campaign_summary.sort_values(
                "Revenue",
                ascending=False
            ),
            x="Campaign_Theme",
            y="Revenue",
            text="Revenue",
            color="ROAS",
            labels={
                "Campaign_Theme": "Campaign",
                "Revenue": "Attributed Revenue (RM)",
                "ROAS": "ROAS (x)"
            }
        )

        campaign_chart.update_traces(
            texttemplate="RM %{text:,.0f}",
            textposition="outside"
        )

        campaign_chart.update_layout(
            height=520,
            xaxis_title="Campaign",
            yaxis_title="Attributed Revenue (RM)",
            xaxis_tickangle=-30
        )

        st.plotly_chart(
            campaign_chart,
            use_container_width=True
        )

        st.subheader("Campaign Efficiency Matrix")

        avg_campaign_spend = campaign_summary["Spend"].mean()
        avg_campaign_revenue = campaign_summary["Revenue"].mean()

        campaign_matrix = px.scatter(
            campaign_summary,
            x="Spend",
            y="Revenue",
            size="Conversions",
            text="Campaign_Theme",
            hover_name="Campaign_Theme",
            labels={
                "Spend": "Marketing Spend (RM)",
                "Revenue": "Attributed Revenue (RM)",
                "Conversions": "Conversions"
            }
        )

        campaign_matrix.add_vline(
            x=avg_campaign_spend,
            line_dash="dash"
        )

        campaign_matrix.add_hline(
            y=avg_campaign_revenue,
            line_dash="dash"
        )

        campaign_matrix.update_traces(
            textposition="top center"
        )

        campaign_matrix.update_layout(
            height=520
        )

        st.plotly_chart(
            campaign_matrix,
            use_container_width=True
        )

        st.subheader("Campaign Performance Table")
       
        campaign_display = campaign_summary.copy()

        campaign_display["Spend"] = campaign_display["Spend"].round(0)
        campaign_display["Revenue"] = campaign_display["Revenue"].round(0)
        campaign_display["ROAS"] = campaign_display["ROAS"].round(2)
        campaign_display["CPA"] = campaign_display["CPA"].round(0)
        campaign_display["Conversion_Rate"] = (
            campaign_display["Conversion_Rate"].round(2)
        )

        campaign_display = campaign_display.rename(
            columns={
                "Campaign_Theme": "Campaign",
                "Spend": "Spend (RM)",
                "Revenue": "Revenue (RM)",
                "ROAS": "ROAS (x)",
                "CPA": "CPA (RM)",
                "Conversion_Rate": "Conversion Rate (%)"
            }
        )

        st.dataframe(
            campaign_display.sort_values(
                "Revenue (RM)",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )
    
        worst_campaign_roas = campaign_summary.loc[
        campaign_summary["ROAS"].idxmin()
        ]

        st.subheader("Campaign Insights")

        st.info(
            f"""
            **Campaign Performance Summary**

            • **{best_campaign_revenue["Campaign_Theme"]}** generated the highest attributed revenue
            at approximately **RM {best_campaign_revenue["Revenue"]:,.0f}**.

            • **{best_campaign_roas["Campaign_Theme"]}** delivered the strongest marketing efficiency
            with a ROAS of **{best_campaign_roas["ROAS"]:.2f}x**.

            • **{best_campaign_conversion["Campaign_Theme"]}** achieved the highest conversion rate
            at **{best_campaign_conversion["Conversion_Rate"]:.2f}%**.

            • **{worst_campaign_roas["Campaign_Theme"]}** recorded the lowest ROAS at
            **{worst_campaign_roas["ROAS"]:.2f}x**, suggesting that campaign targeting,
            promotional mechanics, creative strategy or budget allocation may require review.
            """
        )
    

    # ── Tab 5: Budget Optimization ───────────────────────────────────────────
    with tab_budget:
        st.caption(
            "Efficiency-weighted reallocation model — same total budget, redistributed toward "
            "higher-return channels. Assumes each channel's revenue-per-dollar holds steady as spend shifts."
        )
        bud, _, projected, lift = budget_recommendation(filt)
        bud_plot = go.Figure()
        bud_plot.add_trace(go.Bar(
            name="Current %", x=bud["Channel_Used"], y=bud["current_pct"] * 100, marker_color=PALETTE[2],
        ))
        bud_plot.add_trace(go.Bar(
            name="Recommended %", x=bud["Channel_Used"], y=bud["recommended_pct"] * 100, marker_color=PALETTE[3],
        ))
        bud_plot.update_layout(
            barmode="group",
            title="Budget mix: current vs efficiency-weighted recommendation",
            yaxis_title="Share of spend (%)", xaxis_title="Channel",
            height=420, legend=dict(orientation="h", y=1.05),
        )
        b1, b2 = st.columns([3, 1])
        with b1:
            st.plotly_chart(bud_plot, use_container_width=True)
        with b2:
            st.metric(
                "Projected revenue lift", f"{lift:+.2f}%",
                help="If spend were reallocated toward higher-efficiency channels at the same total budget.",
            )
            st.metric("Projected revenue", f"RM {projected:,.0f}")

        top_camps = (
            filt.groupby(["Campaign_ID", "Campaign_Type", "Channel_Used"], as_index=False)
            .agg(Revenue=("Revenue", "sum"), Spend=("Spend", "sum"), ROI=("ROI", "mean"))
            .sort_values("Revenue", ascending=False)
            .head(15)
        )
        st.subheader("Top campaigns by revenue")
        st.dataframe(top_camps, use_container_width=True, hide_index=True)

        csv_bytes = filt.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download filtered data as CSV",
            data=csv_bytes,
            file_name="campaigniq_filtered.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
