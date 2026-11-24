"""
Cleaning & feature-engineering pipeline.

raw CSVs  ->  validated, typed, de-duplicated tables  ->  data/processed/*.csv.gz
Every rule is logged so the data-quality report can show exactly what changed.

Run: python src/clean.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

CITY_FIX = {
    "bangalore": "Bengaluru", "blr": "Bengaluru", "bengaluru": "Bengaluru",
    "bombay": "Mumbai", "mumbai": "Mumbai",
    "new delhi": "Delhi", "delhi ncr": "Delhi", "delhi": "Delhi",
    "calcutta": "Kolkata", "madras": "Chennai",
}
CITY_META = {
    "Mumbai": (1, "West"), "Delhi": (1, "North"), "Bengaluru": (1, "South"), "Hyderabad": (1, "South"),
    "Chennai": (1, "South"), "Kolkata": (1, "East"), "Pune": (1, "West"), "Ahmedabad": (2, "West"),
    "Jaipur": (2, "North"), "Lucknow": (2, "North"), "Kochi": (2, "South"), "Indore": (2, "West"),
    "Bhubaneswar": (2, "East"), "Coimbatore": (2, "South"), "Nagpur": (2, "West"), "Patna": (3, "East"),
    "Guwahati": (3, "East"), "Vijayawada": (3, "South"), "Ranchi": (3, "East"), "Dehradun": (3, "North"),
    "Madurai": (3, "South"), "Raipur": (3, "East"),
}
PAYMENT_FIX = {
    "upi": "UPI", "u.p.i": "UPI", "cod": "COD", "cash on delivery": "COD",
    "card": "Credit/Debit Card", "credit card": "Credit/Debit Card", "credit/debit card": "Credit/Debit Card",
    "wallet": "Wallet", "paytm wallet": "Wallet", "net banking": "Net Banking",
}


def parse_mixed_dates(s: pd.Series) -> pd.Series:
    """Rows come from two systems: ISO 'YYYY-MM-DD HH:MM:SS' and Indian 'DD/MM/YYYY HH:MM'."""
    iso = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    dmy = pd.to_datetime(s, format="%d/%m/%Y %H:%M", errors="coerce")
    return iso.fillna(dmy)


def clean_orders(log: dict) -> pd.DataFrame:
    o = pd.read_csv(RAW / "orders_raw.csv")
    log["orders_raw_rows"] = len(o)

    before = len(o)
    o = o.drop_duplicates()
    log["duplicate_rows_removed"] = before - len(o)

    o["order_date"] = parse_mixed_dates(o["order_date"])
    log["dates_unparseable"] = int(o["order_date"].isna().sum())
    o = o.dropna(subset=["order_date"])

    o["unit_price"] = (o["unit_price"].astype(str)
                       .str.replace("₹", "", regex=False).str.replace(",", "", regex=False)
                       .astype(float))

    log["quantity_non_positive_removed"] = int((o["quantity"] <= 0).sum())
    o = o[o["quantity"] > 0]
    # Outliers: retail D2C orders above 20 units/line are data-entry errors (IQR fence is ~4)
    q1, q3 = o["quantity"].quantile([0.25, 0.75])
    fence = max(20, q3 + 10 * (q3 - q1))
    log["quantity_outliers_removed"] = int((o["quantity"] > fence).sum())
    o = o[o["quantity"] <= fence]

    original = o["payment_method"].astype("string")
    raw_pay = original.str.strip().str.lower()
    o["payment_method"] = raw_pay.map(PAYMENT_FIX).fillna(original.str.strip())
    log["payment_method_standardised"] = int((original.notna() & (original != o["payment_method"])).sum())
    log["payment_method_missing_filled"] = int(o["payment_method"].isna().sum())
    o["payment_method"] = o["payment_method"].fillna("Unknown")

    o["discount_pct"] = o["discount_pct"].clip(0, 0.9)
    return o


def clean_customers(log: dict) -> pd.DataFrame:
    c = pd.read_csv(RAW / "customers_raw.csv", parse_dates=["signup_date"])
    log["customers_raw_rows"] = len(c)
    key = c["city"].astype("string").str.strip().str.lower()
    fixed = key.map(CITY_FIX)
    log["city_spellings_standardised"] = int((fixed.notna() & (fixed != c["city"])).sum())
    c["city"] = fixed.fillna(c["city"].astype("string").str.strip())
    log["city_missing"] = int(c["city"].isna().sum())
    c["city"] = c["city"].fillna("Unknown")
    c["city_tier"] = c["city"].map(lambda x: CITY_META.get(x, (np.nan, None))[0])
    c["region"] = c["city"].map(lambda x: CITY_META.get(x, (np.nan, "Unknown"))[1])
    log["age_missing_imputed_median"] = int(c["age"].isna().sum())
    c["age"] = c["age"].fillna(c["age"].median()).astype(int)
    c["age_band"] = pd.cut(c["age"], [17, 24, 34, 44, 54, 99], labels=["18-24", "25-34", "35-44", "45-54", "55+"])
    return c


def build_fact(o: pd.DataFrame, c: pd.DataFrame, p: pd.DataFrame, log: dict) -> pd.DataFrame:
    # referential integrity
    orphans = ~o["customer_id"].isin(c["customer_id"]) | ~o["product_id"].isin(p["product_id"])
    log["orphan_lines_removed"] = int(orphans.sum())
    o = o[~orphans]

    f = (o.merge(p, on="product_id", how="left", validate="many_to_one")
          .merge(c, on="customer_id", how="left", validate="many_to_one"))
    f["gross_amount"] = f["quantity"] * f["unit_price"]
    f["net_revenue"] = (f["gross_amount"] * (1 - f["discount_pct"])).round(2)
    f["cogs"] = (f["quantity"] * f["unit_cost"]).round(2)
    f["gross_profit"] = f["net_revenue"] - f["cogs"]
    # returned lines earn no revenue; we eat reverse-logistics cost (~₹80/line)
    f["realised_revenue"] = np.where(f["is_returned"] == 1, 0.0, f["net_revenue"])
    f["realised_profit"] = np.where(f["is_returned"] == 1, -80.0, f["gross_profit"])
    f["order_month"] = f["order_date"].dt.to_period("M").dt.to_timestamp()
    f["order_dow"] = f["order_date"].dt.day_name()
    f["order_hour"] = f["order_date"].dt.hour
    f["discount_band"] = pd.cut(f["discount_pct"], [-0.01, 0.0, 0.10, 0.20, 0.30, 0.9],
                                labels=["0%", "1-10%", "11-20%", "21-30%", ">30%"])
    first = f.groupby("customer_id")["order_date"].transform("min")
    f["is_first_order"] = (f["order_date"] == first).astype(int)
    f["cohort_month"] = first.dt.to_period("M").dt.to_timestamp()
    log["fact_rows_final"] = len(f)
    log["rows_retained_pct"] = round(100 * len(f) / log["orders_raw_rows"], 2)
    return f


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    log: dict = {}
    p = pd.read_csv(RAW / "products.csv")
    o = clean_orders(log)
    c = clean_customers(log)
    f = build_fact(o, c, p, log)

    # assertions = lightweight data contract
    assert f["order_id"].notna().all() and (f["quantity"] > 0).all()
    assert f["net_revenue"].ge(0).all() and f["order_date"].between("2023-01-01", "2025-07-01").all()
    assert not f.duplicated().any()

    f.to_csv(PROC / "fact_order_lines.csv.gz", index=False, compression="gzip")
    c.to_csv(PROC / "dim_customers.csv.gz", index=False, compression="gzip")
    p.to_csv(PROC / "dim_products.csv.gz", index=False, compression="gzip")
    (REPORTS / "data_quality.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
