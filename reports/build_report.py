"""
Build the Technical Contribution Report (PDF) from reports/metrics.json + figures.

Run:  python reports/build_report.py [--author "Your Name"] [--repo URL] [--demo URL]
Out:  reports/Technical_Contribution_Report.pdf
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
INDIGO = colors.HexColor("#4F46E5")
SLATE = colors.HexColor("#334155")
LIGHT = colors.HexColor("#EEF2FF")


def rs(x: float) -> str:
    if abs(x) >= 1e7:
        return f"Rs {x / 1e7:.2f} Cr"
    if abs(x) >= 1e5:
        return f"Rs {x / 1e5:.2f} L"
    return f"Rs {x:,.0f}"


def architecture_png() -> Path:
    out = FIG / "00_architecture.png"
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.set_xlim(0, 11); ax.set_ylim(0, 3.4); ax.axis("off")
    boxes = [
        (0.1, "1. Data generation", "generate_data.py\n12k customers, 122 SKUs\n+ injected dirt", "#64748B"),
        (2.25, "2. Cleaning / ETL", "clean.py\ndedupe, parse, standardise\nvalidate, star schema", "#4F46E5"),
        (4.4, "3. SQL layer", "run_sql.py (SQLite)\nwindow functions, CTEs\n7 business queries", "#06B6D4"),
        (6.55, "4. Analytics & ML", "analysis.py\ncohorts, RFM, K-Means\nlogit, Holt-Winters", "#10B981"),
        (8.7, "5. Delivery", "Streamlit dashboard\nPDF report\nmetrics.json", "#F59E0B"),
    ]
    for x, title, body, col in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.5), 2.0, 2.3, boxstyle="round,pad=0.05,rounding_size=0.15",
                                    fc=col, ec="none", alpha=0.95))
        ax.text(x + 1.0, 2.45, title, ha="center", va="center", color="white", fontsize=11, weight="bold")
        ax.text(x + 1.0, 1.45, body, ha="center", va="center", color="white", fontsize=9, linespacing=1.5)
    for x in [2.12, 4.27, 6.42, 8.57]:
        ax.annotate("", xy=(x + 0.12, 1.65), xytext=(x - 0.04, 1.65),
                    arrowprops=dict(arrowstyle="-|>", color="#0F172A", lw=2))
    ax.text(5.5, 0.15, "data/raw/*.csv  →  data/processed/fact_order_lines + dim_customers + dim_products  →  "
            "reports/", ha="center", fontsize=9, color="#334155")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    return out


def table(rows, col_widths=None, header=True):
    t = Table(rows, colWidths=col_widths, hAlign="LEFT")
    style = [("FONT", (0, 0), (-1, -1), "Helvetica", 8.5),
             ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), INDIGO), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                  ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5)]
    t.setStyle(TableStyle(style))
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", default="")
    ap.add_argument("--repo", default="")
    ap.add_argument("--demo", default="")
    a = ap.parse_args()

    m = json.loads((HERE / "metrics.json").read_text(encoding="utf-8"))
    dq = json.loads((HERE / "data_quality.json").read_text(encoding="utf-8"))
    k = m["kpi"]; fc = m["forecast"]; rm = m["return_model"]; ret = m["retention"]
    arch = architecture_png()

    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=INDIGO, fontSize=15, spaceBefore=10, spaceAfter=6, keepWithNext=1)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=SLATE, fontSize=11.5, spaceBefore=8, spaceAfter=4, keepWithNext=1)
    P = ParagraphStyle("P", parent=ss["BodyText"], fontSize=9.4, leading=13.2, spaceAfter=5)
    B = ParagraphStyle("B", parent=P, leftIndent=12, bulletIndent=2, spaceAfter=2)
    CAP = ParagraphStyle("CAP", parent=P, fontSize=8, textColor=colors.HexColor("#64748B"), alignment=TA_CENTER)
    TITLE = ParagraphStyle("T", parent=ss["Title"], fontSize=22, textColor=colors.HexColor("#0F172A"), leading=27)

    def bullets(items):
        return [Paragraph(i, B, bulletText="•") for i in items]

    def fig(name, width=17, caption=None):
        img = Image(str(FIG / name)); r = img.imageHeight / img.imageWidth
        img.drawWidth = width * cm; img.drawHeight = width * cm * r
        parts = [img]
        if caption:
            parts.append(Paragraph(caption, CAP))
        return parts

    doc = SimpleDocTemplate(str(HERE / "Technical_Contribution_Report.pdf"), pagesize=A4,
                            leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                            title="Retail Sales & Customer Analytics — Technical Contribution Report",
                            author=a.author or "Data Analyst Capstone")
    s = []

    # ---------------- cover / summary
    s.append(Paragraph("Retail Sales &amp; Customer Analytics", TITLE))
    s.append(Paragraph("Technical Contribution Report · Data Analyst Capstone", H2))
    meta = [f"Date: {date.today():%d %B %Y}"]
    if a.author: meta.insert(0, f"Author: {a.author}")
    if a.repo: meta.append(f"Repository: {a.repo}")
    if a.demo: meta.append(f"Live dashboard: {a.demo}")
    s.append(Paragraph(" &nbsp;|&nbsp; ".join(meta), P))
    s.append(Spacer(1, 6))

    s.append(Paragraph("1. Executive summary", H1))
    s.append(Paragraph(
        f"This project analyses <b>{k['order_lines']:,} order lines</b> ({k['orders']:,} orders, "
        f"{k['customers']:,} customers) from an Indian direct-to-consumer e-commerce brand over "
        f"<b>{k['period']}</b>. Raw exports were cleaned through a reproducible Python pipeline, modelled into a "
        "star schema, queried with SQL and analysed with statistical and machine-learning methods. Results are "
        "delivered as an interactive Streamlit dashboard and this report.", P))
    kpi_rows = [["Metric", "Value", "Metric", "Value"],
                ["Realised revenue", rs(k["realised_revenue"]), "Gross margin", f"{k['gross_margin_pct']}%"],
                ["Orders", f"{k['orders']:,}", "Average order value", rs(k["aov"])],
                ["Return rate", f"{k['return_rate_pct']}%", "Repeat customers", f"{k['repeat_customer_pct']}%"],
                ["H1-2025 YoY growth", f"{k['h1_yoy_growth_pct']}%", "Avg. rating", f"{k['avg_rating']} / 5"],
                ["M3 cohort retention", f"{ret['m3_avg_pct']}%", "Forecast MAPE (6-mo hold-out)", f"{fc['mape_pct']}%"]]
    s.append(table(kpi_rows, [4.2 * cm, 3.9 * cm, 5.0 * cm, 3.9 * cm]))
    s.append(Spacer(1, 6))
    disc = {d["discount_band"]: d for d in m["discount_bands"]}
    cod = next(p for p in m["payment"] if p["payment_method"] == "COD")
    upi = next(p for p in m["payment"] if p["payment_method"] == "UPI")
    s.append(Paragraph("<b>Key findings</b>", P))
    s += bullets([
        f"<b>Discounts above 20% destroy value:</b> profit per line falls from Rs {disc['0%']['profit_per_line']:.0f} "
        f"at 0% discount to Rs {disc['21-30%']['profit_per_line']:.0f} at 21–30% and "
        f"Rs {disc['>30%']['profit_per_line']:.0f} beyond 30%, while returns rise to "
        f"{100 * disc['>30%']['return_rate']:.1f}%.",
        f"<b>COD is the biggest controllable return driver:</b> COD lines return at "
        f"{100 * cod['return_rate']:.1f}% vs {100 * upi['return_rate']:.1f}% for UPI. Prepaid methods carry "
        f"roughly half the odds of a return in the logistic model.",
        f"<b>Revenue is concentrated:</b> the top 20% of customers produce "
        f"{m['pareto_top20_customer_revenue_share_pct']}% of revenue; 'Champions' are "
        f"{m['rfm_segments'][0]['customer_share_pct']:.0f}% of customers but "
        f"{m['rfm_segments'][0]['revenue_share_pct']:.0f}% of revenue.",
        "<b>Channel quality differs sharply:</b> " + ", ".join(
            f"{c} {v:.0f}%" for c, v in list(m["repeat_within_3m_by_channel_pct"].items())[:2]) +
        " of new customers re-order within 3 months, vs " + ", ".join(
            f"{c} {v:.0f}%" for c, v in list(m["repeat_within_3m_by_channel_pct"].items())[-2:]) + ".",
        f"<b>Forecast:</b> a {fc['model']} model beat the seasonal-naive baseline "
        f"({fc['mape_pct']}% vs {fc['seasonal_naive_mape_pct']}% MAPE) and projects "
        f"{rs(fc['next_6m_total'])} for Jul–Dec 2025, peaking in the October festive season.",
    ])

    # ---------------- architecture
    s.append(Paragraph("2. Architecture &amp; tech stack", H1))
    s.append(KeepTogether(fig("00_architecture.png", 17.4, "Figure 1 — Pipeline architecture")))
    s.append(Spacer(1, 4))
    s.append(table([
        ["Layer", "Tools", "Responsibility"],
        ["Ingestion / ETL", "Python 3.11, pandas, NumPy", "Typed parsing, de-duplication, standardisation, validation"],
        ["Storage / SQL", "SQLite, SQL (CTEs, window functions)", "Star schema: 1 fact table + 2 dimensions, 7 queries"],
        ["Statistics / ML", "scikit-learn, statsmodels", "K-Means, logistic regression, Holt-Winters, silhouette"],
        ["Visualisation", "Matplotlib, Seaborn, Plotly", "Static report charts + interactive charts"],
        ["Delivery", "Streamlit, ReportLab", "Filterable dashboard, auto-generated PDF report"],
    ], [3.4 * cm, 5.6 * cm, 8.0 * cm]))
    s.append(Paragraph(
        "Every artefact is reproducible with one command (<font face='Courier'>python run_pipeline.py</font>). "
        "The generator is seeded, so the same numbers are produced on every machine.", P))

    # ---------------- data & cleaning
    s.append(Paragraph("3. Data &amp; cleaning", H1))
    s.append(Paragraph(
        "<b>Source.</b> Real customer-level transaction data is confidential, so the dataset is <b>synthetic</b>, "
        "produced by a seeded simulator that encodes realistic behaviour: Diwali/festive seasonality by thinning, "
        "city-tier delivery times, payment-method preferences (COD-heavy in Tier-3), channel-dependent loyalty and "
        "churn, category margins and return rates, and deeper discounts during sales. The simulator then injects "
        "realistic data-quality defects that the cleaning pipeline must repair. Business findings therefore "
        "demonstrate the method; they are not claims about a real company.", P))
    s.append(table([
        ["Cleaning rule", "Rows affected"],
        ["Exact duplicate rows removed (double-ingested batch)", f"{dq['duplicate_rows_removed']:,}"],
        ["Mixed date formats parsed (ISO + DD/MM/YYYY), unparseable", f"{dq['dates_unparseable']:,}"],
        ["Currency strings ('Rs 1,299.00' style) converted to numeric", "all price rows"],
        ["Zero / negative quantities removed", f"{dq['quantity_non_positive_removed']:,}"],
        ["Quantity outliers removed (> 20 units per line)", f"{dq['quantity_outliers_removed']:,}"],
        ["Payment-method spellings standardised", f"{dq['payment_method_standardised']:,}"],
        ["Payment method missing → 'Unknown'", f"{dq['payment_method_missing_filled']:,}"],
        ["City spellings standardised (Bangalore→Bengaluru, Bombay→Mumbai …)", f"{dq['city_spellings_standardised']:,}"],
        ["City missing → 'Unknown' (tier/region left null)", f"{dq['city_missing']:,}"],
        ["Age missing → median imputation", f"{dq['age_missing_imputed_median']:,}"],
        ["Orphan lines (no matching customer/product)", f"{dq['orphan_lines_removed']:,}"],
        ["Final fact rows / retained", f"{dq['fact_rows_final']:,}  ({dq['rows_retained_pct']}%)"],
    ], [12.2 * cm, 4.8 * cm]))
    s.append(Paragraph(
        "Derived fields: net revenue after discount, COGS, gross profit, <i>realised</i> revenue/profit (returned "
        "lines earn nothing and cost Rs 80 reverse logistics), discount bands, cohort month, first-order flag, "
        "weekday/hour. A lightweight data contract (assertions on keys, ranges, duplicates) stops the pipeline "
        "if any rule is violated.", P))

    # ---------------- findings
    s.append(KeepTogether([Paragraph("4. Analysis &amp; findings", H1),
                           Paragraph("4.1 Sales trend and category mix", H2), *fig("01_monthly_revenue.png", 16.5, "Figure 2 — Monthly realised revenue and gross margin")]))
    s.append(Paragraph(
        f"Revenue grew from {rs(m['revenue_by_year']['2023'])} in 2023 to {rs(m['revenue_by_year']['2024'])} in "
        f"2024 and H1-2025 is up {k['h1_yoy_growth_pct']}% year-on-year. October–November spikes coincide with "
        "festive sales, but margin dips in those months because of deeper discounting.", P))
    s.append(KeepTogether(fig("02_category_payment.png", 16.5, "Figure 3 — Category revenue/margin and return rate by payment")))
    top = m["category"][0]; fash = next(c for c in m["category"] if c["category"] == "Fashion")
    s.append(Paragraph(
        f"{top['category']} is the largest category ({top['revenue_share_pct']:.0f}% of revenue) but earns only "
        f"{top['margin_pct']:.0f}% margin. Fashion has strong margin ({fash['margin_pct']:.0f}%) but the highest "
        f"return rate ({100 * fash['return_rate']:.0f}%).", P))

    s.append(KeepTogether([Paragraph("4.2 Cohort retention", H2), *fig("03_cohort_retention.png", 12.5, "Figure 4 — Monthly cohort retention heat-map")]))
    s.append(Paragraph(
        f"Of customers who signed up from January 2023 ({m['cohort_customers']:,}), an average of "
        f"{ret['m1_avg_pct']}% are active one month after their first order, {ret['m3_avg_pct']}% at month 3, "
        f"{ret['m6_avg_pct']}% at month 6 and {ret['m12_avg_pct']}% at month 12. Customers acquired before "
        "2023 are excluded because their true first order is outside the data (left-censoring).", P))

    s.append(KeepTogether([Paragraph("4.3 Customer segmentation (RFM + K-Means)", H2), *fig("04_rfm_kmeans.png", 16.5, "Figure 5 — RFM segment shares and K-Means silhouette scores")]))
    prof = m["kmeans"]["profiles"]
    s.append(Paragraph(
        "Customers were scored 1–5 on Recency, Frequency and Monetary value (quintiles) and mapped to six "
        "rule-based segments. Independently, K-Means was fitted on log-transformed, standardised RFM features; "
        f"k = {m['kmeans']['chosen_k']} gave the best silhouette score "
        f"({m['kmeans']['silhouette_by_k'][str(m['kmeans']['chosen_k'])]}).", P))
    s.append(table([["Cluster", "Customers", "Avg recency (days)", "Avg orders", "Avg revenue"]] +
                   [[p["cluster_name"], f"{p['customers']:,}", f"{p['recency']:.0f}", f"{p['frequency']:.1f}",
                     rs(p["monetary"])] for p in sorted(prof, key=lambda x: -x["monetary"])],
                   [5.4 * cm, 2.4 * cm, 3.4 * cm, 2.4 * cm, 3.4 * cm]))
    at_risk = next(x for x in m["rfm_segments"] if x["rfm_segment"] == "At Risk")
    s.append(Paragraph(
        f"The 'At Risk' segment ({at_risk['customers']:,} customers, {at_risk['revenue_share_pct']:.0f}% of "
        "historical revenue) bought often but has not ordered for over a year — the highest-value win-back "
        "target.", P))

    s.append(KeepTogether([Paragraph("4.4 What drives returns?", H2), *fig("05_return_drivers.png", 14.5, "Figure 6 — Logistic-regression odds ratios for a returned line")]))
    s.append(Paragraph(
        f"A class-balanced logistic regression (one-hot categories + scaled delivery days and price) reaches "
        f"ROC-AUC <b>{rm['roc_auc']:.2f}</b> on a stratified 25% test set ({rm['test_rows']:,} lines). Baselines "
        "are COD, Beauty, 0% discount and Tier-1. Fashion multiplies the odds of a return by "
        f"{rm['top_odds_ratios']['cat__category_Fashion']:.1f}×; discounts above 20% by about "
        f"{rm['top_odds_ratios']['cat__discount_band_21-30%']:.1f}×; every prepaid method roughly halves the odds "
        "versus COD. The model is used for explanation, not as a production classifier.", P))

    s.append(KeepTogether([Paragraph("4.5 Discount economics", H2), *fig("06_discount_economics.png", 13.5, "Figure 7 — Profit per line and return rate by discount band")]))
    s.append(table([["Discount band", "Lines", "Margin", "Profit / line", "Return rate"]] +
                   [[d["discount_band"], f"{d['lines']:,}", f"{d['margin_pct']:.1f}%", f"Rs {d['profit_per_line']:.0f}",
                     f"{100 * d['return_rate']:.1f}%"] for d in m["discount_bands"]],
                   [3.4 * cm, 2.6 * cm, 2.6 * cm, 3.0 * cm, 3.0 * cm]))

    s.append(KeepTogether([Paragraph("4.6 Revenue forecast", H2), *fig("07_forecast.png", 16.5, "Figure 8 — Hold-out validation and 6-month forecast")]))
    s.append(table([["Model (trained Jan 2023 – Dec 2024, tested Jan–Jun 2025)", "Hold-out MAPE"]] +
                   [[n, f"{v}%"] for n, v in sorted(fc["model_comparison_mape_pct"].items(), key=lambda x: x[1])],
                   [12.2 * cm, 4.8 * cm]))
    s.append(Paragraph(
        f"The selected model is refitted on all 30 months to forecast {rs(fc['next_6m_total'])} for Jul–Dec 2025. "
        "With only 2.5 seasonal cycles the forecast should be treated as a planning range and refreshed monthly.", P))

    # ---------------- recommendations
    s.append(Paragraph("5. Recommendations", H1))
    s += bullets([
        "<b>Cap routine discounts at 20%</b> and reserve deeper offers for clearance; model shows >20% bands are "
        "loss-making after returns.",
        "<b>Nudge COD to prepaid</b> (e.g. small UPI cashback, COD fee in Tier-3) — the cheapest lever on returns.",
        "<b>Fix Fashion returns</b> with size guides, fit reviews and try-and-buy pilots before scaling ad spend.",
        "<b>Shift acquisition budget</b> toward Referral and Organic (highest 3-month repeat and 12-month value) and "
        "hold Marketplace / Paid Social to a CAC ceiling.",
        "<b>Run a win-back programme for 'At Risk' customers</b> and protect 'Champions' with loyalty perks.",
        "<b>Plan festive inventory</b> from the forecast: October is the projected peak month.",
    ])

    s.append(Paragraph("6. Limitations &amp; next steps", H1))
    s += bullets([
        "Synthetic data: the pipeline and methods transfer directly to a real export, but the numbers are illustrative.",
        "No marketing-spend data, so CAC and ROAS cannot be computed; channel value is revenue-based only.",
        "Logistic regression is linear in log-odds; a gradient-boosted model with SHAP could capture interactions.",
        "Next: schedule the pipeline (Airflow / GitHub Actions), move storage to Postgres/BigQuery, add A/B test "
        "analysis for the discount cap.",
    ])
    s.append(Paragraph("7. Reproducibility", H1))
    s.append(Paragraph(
        "<font face='Courier'>pip install -r requirements.txt &nbsp;→&nbsp; python run_pipeline.py &nbsp;→&nbsp; "
        "streamlit run app.py</font><br/>All numbers in this report are read from "
        "<font face='Courier'>reports/metrics.json</font>, which the pipeline regenerates.", P))

    def footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5); canvas.setFillColor(colors.HexColor("#94A3B8"))
        canvas.drawString(1.8 * cm, 1.0 * cm, "Retail Sales & Customer Analytics — Technical Contribution Report")
        canvas.drawRightString(A4[0] - 1.8 * cm, 1.0 * cm, f"Page {_doc.page}")
        canvas.restoreState()

    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    out = HERE / "Technical_Contribution_Report.pdf"
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
