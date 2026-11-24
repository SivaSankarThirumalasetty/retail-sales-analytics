# Capstone submission: copy-paste answers

## 1. GitHub repository URL

```
https://github.com/SivaSankarThirumalasetty/retail-sales-analytics
```

## 2. Live demo / deployment URL

```
https://siva-retail-sales-analytics.streamlit.app
```

## 3. Technical contribution report (PDF)

Upload: `reports/Technical_Contribution_Report.pdf` (about 630 KB, 6 pages)

To put your name and links on the cover, run this and then upload the new file:

```
python reports/build_report.py --author "Your Name" --repo "https://github.com/SivaSankarThirumalasetty/retail-sales-analytics" --demo "https://siva-retail-sales-analytics.streamlit.app"
```

## 4. Project architecture & technical description

```
Retail Sales & Customer Analytics: an end-to-end analysis of ~60,000 order lines (35.9k orders, 10.2k customers, Jan 2023 – Jun 2025) for an Indian D2C e-commerce brand.

ARCHITECTURE: 5-stage reproducible Python pipeline (one command: python run_pipeline.py)
1) Seeded data simulator that models festive seasonality, city tiers, COD behaviour, channel loyalty and churn, and injects real-world defects.
2) pandas ETL: removed 902 duplicates, parsed two date formats, converted "₹1,299.00" strings, removed invalid quantities and outliers, standardised 3.3k payment and 784 city spellings, imputed missing values, checked referential integrity, and ran data-contract assertions. Builds a star schema (1 fact + 2 dimension tables) and keeps 97.9% of rows.
3) SQL layer (SQLite): 7 business queries using CTEs and window functions (LAG, RANK, SUM OVER PARTITION).
4) Analytics/ML: KPI and margin analysis, monthly cohort retention, RFM segmentation, K-Means clustering (k chosen by silhouette), a logistic regression explaining returns (ROC-AUC 0.69, odds ratios), and Holt-Winters forecasting compared with seasonal-naive baselines on a 6-month hold-out.
5) Delivery: a Streamlit dashboard (5 tabs with filters) and an auto-generated PDF report.

TECH STACK: Python 3.11, pandas, NumPy, SQL/SQLite, scikit-learn, statsmodels, Matplotlib, Seaborn, Plotly, Streamlit, ReportLab.

KEY RESULTS / PROBLEMS SOLVED:
• Discounts above 20% are loss-making (profit per line falls from ₹593 to –₹44, and returns rise to 14.7%), so routine discounts should be capped at 20%.
• COD lines return at 16.7% vs 9.6% for UPI; prepaid payment roughly halves the odds of a return.
• Top 20% of customers produce 57% of revenue. Identified 2,214 "At Risk" customers (25% of past revenue) for win-back.
• Referral and Organic customers are worth ~1.8x more in their first 12 months than Marketplace and Paid Social customers.
• Forecast MAPE is 5.85% vs 10.66% for the naive baseline; it projects ₹2.19 Cr for Jul–Dec 2025 with an October peak.

Note: real customer transactions are confidential, so the data is synthetic but behaviour-realistic. The pipeline works on a real export without changes.
```
