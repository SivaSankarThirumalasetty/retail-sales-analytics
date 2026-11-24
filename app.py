"""
Interactive dashboard for the Retail Sales & Customer Analytics project.
Run locally:  streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
COLORS = ["#4F46E5", "#10B981", "#F59E0B", "#EF4444", "#06B6D4", "#8B5CF6", "#64748B"]

st.set_page_config(page_title="Retail Sales & Customer Analytics", page_icon="📊", layout="wide")


def inr(x: float) -> str:
    if abs(x) >= 1e7:
        return f"₹{x / 1e7:.2f} Cr"
    if abs(x) >= 1e5:
        return f"₹{x / 1e5:.2f} L"
    return f"₹{x:,.0f}"


@st.cache_data
def load():
    f = pd.read_csv(PROC / "fact_order_lines.csv.gz", parse_dates=["order_date", "order_month", "cohort_month"])
    f["tier_label"] = f["city_tier"].map({1: "Tier 1", 2: "Tier 2", 3: "Tier 3"}).fillna("Unknown")
    rfm = pd.read_csv(PROC / "rfm.csv.gz")
    cohort = pd.read_csv(PROC / "cohort.csv", index_col=0)
    fc = pd.read_csv(PROC / "forecast.csv", parse_dates=["month"])
    metrics = json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))
    dq = json.loads((REPORTS / "data_quality.json").read_text(encoding="utf-8"))
    return f, rfm, cohort, fc, metrics, dq


f, rfm, cohort, fc, metrics, dq = load()

# ------------------------------------------------------------------ sidebar filters
st.sidebar.title("Filters")
dmin, dmax = f.order_date.min().date(), f.order_date.max().date()
date_range = st.sidebar.date_input("Order date", (dmin, dmax), min_value=dmin, max_value=dmax)
cats = st.sidebar.multiselect("Category", sorted(f.category.unique()), default=sorted(f.category.unique()))
TIERS = ["Tier 1", "Tier 2", "Tier 3", "Unknown"]
tiers = st.sidebar.multiselect("City tier", TIERS, default=TIERS)
chans = st.sidebar.multiselect("Acquisition channel", sorted(f.acquisition_channel.unique()),
                               default=sorted(f.acquisition_channel.unique()))
st.sidebar.caption("Dataset: synthetic but behaviour-realistic Indian D2C e-commerce data "
                   "(Jan 2023 – Jun 2025). See README for how it was generated and cleaned.")

start, end = (date_range if isinstance(date_range, tuple) and len(date_range) == 2 else (dmin, dmax))
mask = (f.order_date.dt.date.between(start, end) & f.category.isin(cats)
        & f.tier_label.isin(tiers) & f.acquisition_channel.isin(chans))
d = f[mask]
if d.empty:
    st.warning("No data for the selected filters.")
    st.stop()

st.title("📊 Retail Sales & Customer Analytics")
st.caption("End-to-end analysis of 60k order lines: data cleaning → KPIs → cohorts → RFM & K-Means → "
           "return-risk modelling → discount economics → revenue forecasting")

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Realised revenue", inr(d.realised_revenue.sum()))
k2.metric("Orders", f"{d.order_id.nunique():,}")
k3.metric("AOV", f"₹{d.realised_revenue.sum() / d.order_id.nunique():,.0f}")
k4.metric("Gross margin", f"{100 * d.realised_profit.sum() / max(d.realised_revenue.sum(), 1):.1f}%")
k5.metric("Return rate", f"{100 * d.is_returned.mean():.1f}%")
k6.metric("Customers", f"{d.customer_id.nunique():,}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Sales overview", "Customers & retention", "Returns & discounts",
                                         "Forecast", "Data quality & method"])

# ------------------------------------------------------------------ overview
with tab1:
    m = d.groupby("order_month").agg(revenue=("realised_revenue", "sum"), profit=("realised_profit", "sum"),
                                     orders=("order_id", "nunique")).reset_index()
    m["margin"] = 100 * m.profit / m.revenue
    fig = go.Figure()
    fig.add_bar(x=m.order_month, y=m.revenue, name="Revenue", marker_color=COLORS[0])
    fig.add_scatter(x=m.order_month, y=m.margin, name="Gross margin %", yaxis="y2",
                    line=dict(color=COLORS[2], width=3))
    fig.update_layout(title="Monthly realised revenue & margin", height=380,
                      yaxis=dict(title="₹"), yaxis2=dict(title="%", overlaying="y", side="right", showgrid=False),
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    cat = d.groupby("category").agg(revenue=("realised_revenue", "sum"), profit=("realised_profit", "sum"),
                                    returns=("is_returned", "mean")).reset_index()
    cat["margin_pct"] = 100 * cat.profit / cat.revenue
    c1.plotly_chart(px.scatter(cat, x="revenue", y="margin_pct", size="revenue", color="category",
                               text="category", color_discrete_sequence=COLORS,
                               title="Category: revenue vs margin (bubble = revenue)")
                    .update_traces(textposition="top center").update_layout(showlegend=False, height=400),
                    use_container_width=True)
    city = (d[d.city != "Unknown"].groupby(["region", "city"]).realised_revenue.sum().reset_index())
    c2.plotly_chart(px.treemap(city, path=["region", "city"], values="realised_revenue",
                               color_discrete_sequence=COLORS, title="Revenue by region → city")
                    .update_layout(height=400), use_container_width=True)

    c3, c4 = st.columns(2)
    heat = d.groupby(["order_dow", "order_hour"]).order_id.nunique().reset_index()
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heat = heat.pivot(index="order_dow", columns="order_hour", values="order_id").reindex(days)
    c3.plotly_chart(px.imshow(heat, color_continuous_scale="Blues", aspect="auto",
                              title="Orders by weekday × hour").update_layout(height=360),
                    use_container_width=True)
    ch = d.groupby("acquisition_channel").agg(revenue=("realised_revenue", "sum"),
                                               customers=("customer_id", "nunique")).reset_index()
    ch["revenue_per_customer"] = ch.revenue / ch.customers
    c4.plotly_chart(px.bar(ch.sort_values("revenue_per_customer"), x="revenue_per_customer",
                           y="acquisition_channel", orientation="h", color_discrete_sequence=[COLORS[1]],
                           title="Revenue per customer by acquisition channel").update_layout(height=360),
                    use_container_width=True)

# ------------------------------------------------------------------ customers
with tab2:
    c1, c2 = st.columns([1.1, 1])
    r = rfm[rfm.customer_id.isin(d.customer_id.unique())]
    seg = r.groupby("rfm_segment").agg(customers=("customer_id", "size"), revenue=("monetary", "sum")).reset_index()
    seg["% customers"] = 100 * seg.customers / seg.customers.sum()
    seg["% revenue"] = 100 * seg.revenue / seg.revenue.sum()
    long = seg.melt(id_vars="rfm_segment", value_vars=["% customers", "% revenue"])
    c1.plotly_chart(px.bar(long, x="value", y="rfm_segment", color="variable", barmode="group", orientation="h",
                           color_discrete_sequence=[COLORS[6], COLORS[0]],
                           title="RFM segments — share of customers vs revenue").update_layout(height=400),
                    use_container_width=True)
    c2.plotly_chart(px.scatter(r.sample(min(3000, len(r)), random_state=1), x="recency", y="monetary",
                               color="cluster_name", size="frequency", log_y=True, opacity=0.6,
                               color_discrete_sequence=COLORS, title="K-Means clusters on RFM (sample)")
                    .update_layout(height=400, legend=dict(orientation="h", y=-0.25)), use_container_width=True)
    st.plotly_chart(px.imshow(cohort.drop(columns=["0"], errors="ignore"), color_continuous_scale="Blues",
                              aspect="auto", labels=dict(x="Months since first order", y="Cohort", color="% active"),
                              title="Cohort retention (customers who signed up from Jan 2023)")
                    .update_layout(height=560), use_container_width=True)
    st.info(f"Average retention: M1 {metrics['retention']['m1_avg_pct']}% · M3 {metrics['retention']['m3_avg_pct']}% · "
            f"M6 {metrics['retention']['m6_avg_pct']}% · M12 {metrics['retention']['m12_avg_pct']}%.  "
            f"Top 20% of customers generate {metrics['pareto_top20_customer_revenue_share_pct']}% of revenue.")

# ------------------------------------------------------------------ returns
with tab3:
    c1, c2 = st.columns(2)
    pay = d[d.payment_method != "Unknown"].groupby(["payment_method"]).is_returned.mean().mul(100).reset_index()
    c1.plotly_chart(px.bar(pay.sort_values("is_returned"), x="is_returned", y="payment_method", orientation="h",
                           color=pay.sort_values("is_returned").payment_method.eq("COD"),
                           color_discrete_map={True: COLORS[3], False: COLORS[6]},
                           labels={"is_returned": "Return rate %"}, title="Return rate by payment method")
                    .update_layout(showlegend=False, height=360), use_container_width=True)
    disc = d.groupby("discount_band", observed=True).agg(profit_per_line=("realised_profit", "mean"),
                                                         return_rate=("is_returned", "mean")).reset_index()
    fig = go.Figure()
    fig.add_bar(x=disc.discount_band, y=disc.profit_per_line, name="Profit per line (₹)", marker_color=COLORS[0])
    fig.add_scatter(x=disc.discount_band, y=100 * disc.return_rate, name="Return rate %", yaxis="y2",
                    line=dict(color=COLORS[3], width=3))
    fig.update_layout(title="Discount economics", height=360,
                      yaxis2=dict(overlaying="y", side="right", showgrid=False), legend=dict(orientation="h", y=1.12))
    c2.plotly_chart(fig, use_container_width=True)
    rm = metrics["return_model"]
    odds = pd.Series({**rm["top_odds_ratios"], **rm["lowest_odds_ratios"]}).sort_values()
    odds.index = [i.replace("cat__", "").replace("num__", "").replace("_", " ") for i in odds.index]
    st.plotly_chart(px.bar(x=odds.values, y=odds.index, orientation="h", log_x=True,
                           color=odds.values > 1, color_discrete_map={True: COLORS[3], False: COLORS[1]},
                           labels={"x": "Odds ratio (log scale)", "y": ""},
                           title=f"Return-risk drivers — logistic regression (ROC-AUC {rm['roc_auc']:.2f}, "
                                 f"baselines: COD, Beauty, 0% discount, Tier 1)")
                    .update_layout(showlegend=False, height=420), use_container_width=True)

# ------------------------------------------------------------------ forecast
with tab4:
    fm = metrics["forecast"]
    fig = go.Figure()
    fig.add_scatter(x=fc.month, y=fc.actual, name="Actual", line=dict(color=COLORS[0], width=3))
    fig.add_scatter(x=fc.month, y=fc.holdout_pred, name=f"Hold-out prediction (MAPE {fm['mape_pct']}%)",
                    line=dict(color=COLORS[2], dash="dash"))
    fig.add_scatter(x=fc.month, y=fc.forecast, name="Next 6 months", line=dict(color=COLORS[1], dash="dot", width=3))
    fig.update_layout(title=f"Revenue forecast — {fm['model']}", height=420, yaxis_title="₹",
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)
    comp = pd.Series(fm["model_comparison_mape_pct"]).sort_values().rename("Hold-out MAPE %").to_frame()
    c1, c2 = st.columns(2)
    c1.subheader("Model comparison (Jan–Jun 2025 hold-out)")
    c1.dataframe(comp, use_container_width=True)
    c2.subheader("Forecast Jul–Dec 2025")
    c2.dataframe(pd.Series(fm["next_6m"]).map(inr).rename("Forecast revenue").to_frame(), use_container_width=True)
    c2.metric("Next 6 months total", inr(fm["next_6m_total"]))
    st.caption("Forecast models use the full company data and ignore the sidebar filters.")

# ------------------------------------------------------------------ data quality
with tab5:
    st.subheader("Cleaning log")
    st.dataframe(pd.Series(dq).rename("value").to_frame(), use_container_width=True)
    st.markdown("""
**Pipeline:** `generate_data.py` → `clean.py` (type repair, de-duplication, mixed-format date parsing,
currency-string parsing, city/payment standardisation, outlier & invalid-quantity removal, referential checks,
data-contract assertions) → `analysis.py` (KPIs, cohorts, RFM, K-Means, logistic regression, Holt-Winters)
→ `run_sql.py` (SQLite star schema + 7 business queries) → this dashboard and the PDF report.
""")
    st.download_button("Download filtered order lines (CSV)", d.to_csv(index=False).encode(),
                       "filtered_order_lines.csv", "text/csv")
