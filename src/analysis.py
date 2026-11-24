"""
Analytical layer: KPIs, cohort retention, RFM segmentation, K-Means clustering,
return-risk drivers (logistic regression), discount economics and a
Holt-Winters revenue forecast with hold-out validation.

Outputs:
  reports/metrics.json            headline numbers used by README / PDF / dashboard
  reports/figures/*.png           static charts for the PDF report
  data/processed/rfm.csv.gz       customer-level segments
  data/processed/cohort.csv       retention matrix
  data/processed/forecast.csv     actual vs fitted vs forecast

Run: python src/analysis.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"

PALETTE = ["#4F46E5", "#10B981", "#F59E0B", "#EF4444", "#06B6D4", "#8B5CF6", "#64748B"]
sns.set_theme(style="whitegrid", palette=PALETTE)
plt.rcParams.update({"figure.dpi": 130, "axes.titleweight": "bold", "axes.titlesize": 12})


def inr(x: float) -> str:
    """Format rupees in lakh/crore notation."""
    if abs(x) >= 1e7:
        return f"₹{x / 1e7:.2f} Cr"
    if abs(x) >= 1e5:
        return f"₹{x / 1e5:.2f} L"
    return f"₹{x:,.0f}"


def load() -> pd.DataFrame:
    return pd.read_csv(PROC / "fact_order_lines.csv.gz",
                       parse_dates=["order_date", "order_month", "cohort_month", "signup_date"])


# ---------------------------------------------------------------- KPIs
def kpis(f: pd.DataFrame, m: dict) -> pd.DataFrame:
    orders = f.groupby("order_id").agg(rev=("realised_revenue", "sum"), date=("order_date", "first"))
    m["kpi"] = {
        "net_revenue": round(f["net_revenue"].sum(), 2),
        "realised_revenue": round(f["realised_revenue"].sum(), 2),
        "realised_gross_profit": round(f["realised_profit"].sum(), 2),
        "gross_margin_pct": round(100 * f["realised_profit"].sum() / f["realised_revenue"].sum(), 2),
        "orders": int(f["order_id"].nunique()),
        "order_lines": int(len(f)),
        "customers": int(f["customer_id"].nunique()),
        "aov": round(f["realised_revenue"].sum() / f["order_id"].nunique(), 2),
        "return_rate_pct": round(100 * f["is_returned"].mean(), 2),
        "avg_rating": round(f["rating"].mean(), 2),
        "repeat_customer_pct": round(100 * (f.groupby("customer_id")["order_id"].nunique() > 1).mean(), 2),
        "period": f"{f['order_date'].min():%b %Y} – {f['order_date'].max():%b %Y}",
    }
    monthly = (f.groupby("order_month")
                .agg(revenue=("realised_revenue", "sum"), profit=("realised_profit", "sum"),
                     orders=("order_id", "nunique"), customers=("customer_id", "nunique"),
                     new_customers=("is_first_order", "sum"))
                .reset_index())
    # YoY for H1 2025 vs H1 2024
    h1_25 = monthly[(monthly.order_month >= "2025-01-01")]["revenue"].sum()
    h1_24 = monthly[(monthly.order_month >= "2024-01-01") & (monthly.order_month < "2024-07-01")]["revenue"].sum()
    m["kpi"]["h1_yoy_growth_pct"] = round(100 * (h1_25 / h1_24 - 1), 2)
    fy = monthly.assign(y=monthly.order_month.dt.year).groupby("y")["revenue"].sum()
    m["revenue_by_year"] = {int(k): round(v, 2) for k, v in fy.items()}

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(monthly.order_month, monthly.revenue / 1e5, width=20, color=PALETTE[0], label="Realised revenue")
    ax2 = ax.twinx()
    ax2.plot(monthly.order_month, 100 * monthly.profit / monthly.revenue, color=PALETTE[2], marker="o",
             lw=2, label="Gross margin %")
    ax.set_ylabel("Revenue (₹ lakh)"); ax2.set_ylabel("Gross margin %"); ax2.grid(False)
    ax.set_title("Monthly realised revenue & margin — festive peaks every Oct/Nov")
    fig.legend(loc="upper left", bbox_to_anchor=(0.07, 0.9)); fig.tight_layout()
    fig.savefig(FIG / "01_monthly_revenue.png"); plt.close(fig)
    return monthly


def segments(f: pd.DataFrame, m: dict) -> None:
    cat = (f.groupby("category")
            .agg(revenue=("realised_revenue", "sum"), profit=("realised_profit", "sum"),
                 return_rate=("is_returned", "mean"), lines=("order_id", "size"))
            .sort_values("revenue", ascending=False))
    cat["margin_pct"] = 100 * cat.profit / cat.revenue
    cat["revenue_share_pct"] = 100 * cat.revenue / cat.revenue.sum()
    m["category"] = cat.round(3).reset_index().to_dict(orient="records")

    tier = f.groupby("city_tier").agg(revenue=("realised_revenue", "sum"), return_rate=("is_returned", "mean"),
                                      delivery=("delivery_days", "mean"),
                                      aov=("realised_revenue", "sum")).reset_index()
    tier["aov"] = tier["aov"] / f.groupby("city_tier")["order_id"].nunique().values
    m["city_tier"] = tier.round(3).to_dict(orient="records")

    pay = f.groupby("payment_method").agg(lines=("order_id", "size"), return_rate=("is_returned", "mean")).reset_index()
    m["payment"] = pay.round(4).to_dict(orient="records")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    c2 = cat.sort_values("revenue")
    axes[0].barh(c2.index, c2.revenue / 1e5, color=PALETTE[0])
    axes[0].set_title("Realised revenue by category (₹ lakh)")
    for i, (r, mg) in enumerate(zip(c2.revenue / 1e5, c2.margin_pct)):
        axes[0].text(r, i, f"  {mg:.0f}% margin", va="center", fontsize=8)
    axes[0].set_xlim(0, c2.revenue.max() / 1e5 * 1.35)
    pr = pay[pay.payment_method != "Unknown"].sort_values("return_rate")
    cols = [PALETTE[3] if p == "COD" else PALETTE[6] for p in pr.payment_method]
    axes[1].barh(pr.payment_method, 100 * pr.return_rate, color=cols)
    axes[1].set_title("Return rate by payment method (%)")
    fig.tight_layout(); fig.savefig(FIG / "02_category_payment.png"); plt.close(fig)


# ---------------------------------------------------------------- cohorts
def cohorts(f: pd.DataFrame, m: dict) -> None:
    # Only customers who signed up inside the export window: earlier customers are left-censored
    # (their true first order is not in the data) and would inflate early cohorts.
    f = f[f["signup_date"] >= "2023-01-01"]
    m["cohort_customers"] = int(f["customer_id"].nunique())
    d = f[["customer_id", "order_month", "cohort_month"]].drop_duplicates()
    d["age"] = ((d.order_month.dt.year - d.cohort_month.dt.year) * 12
                + (d.order_month.dt.month - d.cohort_month.dt.month))
    tab = d.pivot_table(index="cohort_month", columns="age", values="customer_id", aggfunc="nunique")
    ret = tab.div(tab[0], axis=0) * 100
    ret = ret.loc[:, :12]
    ret.index = ret.index.strftime("%Y-%m")
    ret.to_csv(PROC / "cohort.csv")
    m["retention"] = {
        "m1_avg_pct": round(ret[1].mean(), 2),
        "m3_avg_pct": round(ret[3].mean(), 2),
        "m6_avg_pct": round(ret[6].mean(), 2),
        "m12_avg_pct": round(ret[12].dropna().mean(), 2),
    }
    # retention by acquisition channel (M3)
    first = f[f.is_first_order == 1].drop_duplicates("customer_id")[["customer_id", "acquisition_channel", "cohort_month"]]
    d2 = d.merge(first[["customer_id", "acquisition_channel"]], on="customer_id")
    eligible = first[first.cohort_month <= "2025-03-01"]
    active3 = d2[(d2.age >= 1) & (d2.age <= 3)].customer_id.unique()
    ch = eligible.assign(ret=eligible.customer_id.isin(active3)).groupby("acquisition_channel")["ret"].mean() * 100
    m["repeat_within_3m_by_channel_pct"] = ch.round(2).sort_values(ascending=False).to_dict()

    fig, ax = plt.subplots(figsize=(11, 6.5))
    sns.heatmap(ret.iloc[:, 1:], cmap="Blues", annot=False, cbar_kws={"label": "% of cohort active"}, ax=ax)
    ax.set_title("Monthly cohort retention (months since first purchase)")
    ax.set_xlabel("Months since first order"); ax.set_ylabel("Cohort (first order month)")
    fig.tight_layout(); fig.savefig(FIG / "03_cohort_retention.png"); plt.close(fig)


# ---------------------------------------------------------------- RFM + KMeans
def rfm(f: pd.DataFrame, m: dict) -> None:
    snap = f["order_date"].max() + pd.Timedelta(days=1)
    r = f.groupby("customer_id").agg(
        recency=("order_date", lambda s: (snap - s.max()).days),
        frequency=("order_id", "nunique"),
        monetary=("realised_revenue", "sum"),
        channel=("acquisition_channel", "first"),
        city_tier=("city_tier", "first"),
    )
    r["R"] = pd.qcut(r.recency, 5, labels=[5, 4, 3, 2, 1]).astype(int)
    r["F"] = pd.qcut(r.frequency.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    r["M"] = pd.qcut(r.monetary.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)

    def label(row) -> str:
        R, F = row.R, row.F
        if R >= 4 and F >= 4: return "Champions"
        if R >= 3 and F >= 3: return "Loyal"
        if R >= 4 and F <= 2: return "New / Promising"
        if R == 3 and F <= 2: return "Need Attention"
        if R <= 2 and F >= 3: return "At Risk"
        return "Hibernating"
    r["rfm_segment"] = r.apply(label, axis=1)

    # K-Means on log-scaled RFM; choose k by silhouette
    X = StandardScaler().fit_transform(np.log1p(r[["recency", "frequency", "monetary"]]))
    sil = {}
    for k in range(2, 8):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        sil[k] = silhouette_score(X, km.labels_, sample_size=4000, random_state=42)
    best_k = max(sil, key=sil.get)
    best_k = best_k if best_k >= 3 else 4  # 2 clusters is not actionable for marketing
    km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X)
    r["cluster"] = km.labels_
    prof = r.groupby("cluster")[["recency", "frequency", "monetary"]].mean()
    # name clusters from their own profile: value tier + recency status
    def cname(row) -> str:
        if row.monetary < 500:
            return "Zero-Value (orders returned)"
        value = ("High-Value" if row.monetary >= 10000 else "Mid-Value" if row.monetary >= 4000 else "Low-Value")
        status = ("Active" if row.recency <= 120 else "Cooling" if row.recency <= 365 else "Lapsed")
        return f"{value} · {status}"
    names = {c: cname(row) for c, row in prof.iterrows()}
    r["cluster_name"] = r.cluster.map(names)
    r.reset_index().to_csv(PROC / "rfm.csv.gz", index=False, compression="gzip")

    seg = r.groupby("rfm_segment").agg(customers=("monetary", "size"), revenue=("monetary", "sum"),
                                       avg_recency=("recency", "mean"), avg_frequency=("frequency", "mean"))
    seg["customer_share_pct"] = 100 * seg.customers / seg.customers.sum()
    seg["revenue_share_pct"] = 100 * seg.revenue / seg.revenue.sum()
    m["rfm_segments"] = seg.round(2).sort_values("revenue", ascending=False).reset_index().to_dict(orient="records")
    m["kmeans"] = {"silhouette_by_k": {k: round(v, 4) for k, v in sil.items()}, "chosen_k": int(best_k),
                   "profiles": r.groupby("cluster_name")[["recency", "frequency", "monetary"]].mean().round(1)
                   .assign(customers=r.groupby("cluster_name").size()).reset_index().to_dict(orient="records")}
    top20 = r.monetary.sort_values(ascending=False)
    m["pareto_top20_customer_revenue_share_pct"] = round(100 * top20.head(int(len(r) * 0.2)).sum() / top20.sum(), 2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    s2 = seg.sort_values("revenue_share_pct")
    y = np.arange(len(s2))
    axes[0].barh(y - 0.2, s2.customer_share_pct, 0.4, label="% customers", color=PALETTE[6])
    axes[0].barh(y + 0.2, s2.revenue_share_pct, 0.4, label="% revenue", color=PALETTE[0])
    axes[0].set_yticks(y, s2.index); axes[0].legend(); axes[0].set_title("RFM segments: customers vs revenue")
    axes[1].plot(list(sil), list(sil.values()), marker="o", color=PALETTE[1])
    axes[1].axvline(best_k, ls="--", color=PALETTE[3]); axes[1].set_title(f"K-Means silhouette (chosen k={best_k})")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("silhouette")
    fig.tight_layout(); fig.savefig(FIG / "04_rfm_kmeans.png"); plt.close(fig)


# ---------------------------------------------------------------- return drivers
def returns_model(f: pd.DataFrame, m: dict) -> None:
    feats_cat = ["payment_method", "category", "discount_band", "city_tier"]
    feats_num = ["delivery_days", "unit_price"]
    d = f[feats_cat + feats_num + ["is_returned"]].dropna().copy()
    d["city_tier"] = d["city_tier"].astype(int).astype(str)
    d["discount_band"] = d["discount_band"].astype(str)
    X_tr, X_te, y_tr, y_te = train_test_split(d[feats_cat + feats_num], d.is_returned, test_size=0.25,
                                              random_state=42, stratify=d.is_returned)
    pre = ColumnTransformer([("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), feats_cat),
                             ("num", StandardScaler(), feats_num)])
    pipe = Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    pipe.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
    names = pipe.named_steps["pre"].get_feature_names_out()
    coefs = pd.Series(pipe.named_steps["lr"].coef_[0], index=names)
    odds = np.exp(coefs).sort_values(ascending=False)
    m["return_model"] = {"roc_auc": round(auc, 4), "test_rows": int(len(X_te)),
                         "top_odds_ratios": odds.head(8).round(3).to_dict(),
                         "lowest_odds_ratios": odds.tail(4).round(3).to_dict()}
    top = pd.concat([odds.head(7), odds.tail(4)])
    top.index = [i.replace("cat__", "").replace("num__", "").replace("_", " ") for i in top.index]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.barh(top.index[::-1], top.values[::-1],
            color=[PALETTE[3] if v > 1 else PALETTE[1] for v in top.values[::-1]])
    ax.axvline(1, color="black", lw=1); ax.set_xscale("log")
    ax.set_title(f"Return-risk drivers — logistic regression odds ratios (ROC-AUC {auc:.2f})")
    fig.tight_layout(); fig.savefig(FIG / "05_return_drivers.png"); plt.close(fig)


# ---------------------------------------------------------------- discounts
def discounts(f: pd.DataFrame, m: dict) -> None:
    d = f.groupby("discount_band", observed=True).agg(
        lines=("order_id", "size"), revenue=("realised_revenue", "sum"), profit=("realised_profit", "sum"),
        return_rate=("is_returned", "mean"))
    d["margin_pct"] = 100 * d.profit / d.revenue
    d["profit_per_line"] = d.profit / d.lines
    m["discount_bands"] = d.round(3).reset_index().astype({"discount_band": str}).to_dict(orient="records")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(d.index.astype(str), d.profit_per_line, color=PALETTE[0])
    ax2 = ax.twinx(); ax2.plot(d.index.astype(str), 100 * d.return_rate, color=PALETTE[3], marker="o")
    ax2.set_ylabel("Return rate %"); ax2.grid(False); ax.set_ylabel("Profit per line (₹)")
    ax.set_title("Deeper discounts erode profit per line and raise returns")
    fig.tight_layout(); fig.savefig(FIG / "06_discount_economics.png"); plt.close(fig)


# ---------------------------------------------------------------- forecast
def forecast(monthly: pd.DataFrame, m: dict) -> None:
    s = monthly.set_index("order_month")["revenue"].asfreq("MS")
    train, test = s.iloc[:-6], s.iloc[-6:]

    def mape_of(pred) -> float:
        return float(np.mean(np.abs((test.values - np.asarray(pred)) / test.values)) * 100)

    # Candidate models, selected on a 6-month hold-out (Jan–Jun 2025)
    candidates = {
        "Holt-Winters (add. trend, mult. season)":
            lambda x: ExponentialSmoothing(x, trend="add", seasonal="mul", seasonal_periods=12,
                                           initialization_method="estimated").fit(),
        "Holt-Winters (damped trend, mult. season)":
            lambda x: ExponentialSmoothing(x, trend="add", damped_trend=True, seasonal="mul",
                                           seasonal_periods=12, initialization_method="estimated").fit(),
        "Holt-Winters (add. trend, add. season)":
            lambda x: ExponentialSmoothing(x, trend="add", seasonal="add", seasonal_periods=12,
                                           initialization_method="estimated").fit(),
    }
    scores, preds = {}, {}
    for name, builder in candidates.items():
        preds[name] = builder(train).forecast(6)
        scores[name] = mape_of(preds[name])
    # baselines
    naive = s.shift(12).iloc[-6:]
    growth = train.iloc[-12:].sum() / train.iloc[-24:-12].sum()
    scores["Seasonal naive (same month last year)"] = mape_of(naive)
    scores["Seasonal naive × YoY growth"] = mape_of(naive * growth)
    best = min(candidates, key=scores.get)
    pred = preds[best]
    mape = scores[best]
    mape_naive = scores["Seasonal naive (same month last year)"]
    fut = candidates[best](s).forecast(6)
    out = pd.DataFrame({"actual": s, "holdout_pred": pred}).join(pd.DataFrame({"forecast": fut}), how="outer")
    out.index.name = "month"
    out.to_csv(PROC / "forecast.csv")
    m["forecast"] = {"model": best, "model_comparison_mape_pct": {k: round(v, 2) for k, v in scores.items()},
                     "holdout_months": 6, "mape_pct": round(mape, 2),
                     "seasonal_naive_mape_pct": round(mape_naive, 2),
                     "next_6m_total": round(float(fut.sum()), 2),
                     "next_6m": {k.strftime("%Y-%m"): round(float(v), 2) for k, v in fut.items()}}
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(s.index, s / 1e5, label="Actual", color=PALETTE[0], lw=2)
    ax.plot(pred.index, pred / 1e5, "--o", label=f"Hold-out prediction (MAPE {mape:.1f}%)", color=PALETTE[2])
    ax.plot(fut.index, fut / 1e5, "--", label="6-month forecast", color=PALETTE[1], lw=2)
    ax.set_ylabel("₹ lakh"); ax.legend(); ax.set_title(f"Revenue forecast — {best}")
    fig.tight_layout(); fig.savefig(FIG / "07_forecast.png"); plt.close(fig)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    FIG.mkdir(parents=True, exist_ok=True)
    f = load()
    m: dict = {}
    monthly = kpis(f, m)
    monthly.to_csv(PROC / "monthly_kpis.csv", index=False)
    segments(f, m)
    cohorts(f, m)
    rfm(f, m)
    returns_model(f, m)
    discounts(f, m)
    forecast(monthly, m)
    (REPORTS / "metrics.json").write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
    k = m["kpi"]
    print(f"Revenue {inr(k['realised_revenue'])} | orders {k['orders']:,} | AOV ₹{k['aov']:.0f} | "
          f"margin {k['gross_margin_pct']}% | returns {k['return_rate_pct']}% | "
          f"M3 retention {m['retention']['m3_avg_pct']}% | AUC {m['return_model']['roc_auc']} | "
          f"MAPE {m['forecast']['mape_pct']}% (naive {m['forecast']['seasonal_naive_mape_pct']}%)")


if __name__ == "__main__":
    main()
