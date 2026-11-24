"""
Generate a realistic (synthetic) raw dataset for an Indian D2C e-commerce brand.

Why synthetic?  Real transactional data with customer IDs is confidential, so the
generator encodes realistic business behaviour (festive seasonality, city tiers,
COD return risk, discount-driven demand, customer churn) and then deliberately
injects the kind of dirt seen in real exports (duplicates, mixed date formats,
currency strings, city spelling variants, missing values, typos, outliers).
The cleaning pipeline in `clean.py` has to repair all of it.

Output: data/raw/orders_raw.csv, data/raw/customers_raw.csv, data/raw/products.csv
Run:    python src/generate_data.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

START = pd.Timestamp("2023-01-01")
END = pd.Timestamp("2025-06-30")
N_CUSTOMERS = 12000
PRE_START = pd.Timestamp("2022-01-01")  # customers acquired before the export window (warm start)

CITIES = {
    # city: (tier, weight, region)
    "Mumbai": (1, 0.11, "West"), "Delhi": (1, 0.11, "North"), "Bengaluru": (1, 0.10, "South"),
    "Hyderabad": (1, 0.08, "South"), "Chennai": (1, 0.07, "South"), "Kolkata": (1, 0.06, "East"),
    "Pune": (1, 0.06, "West"), "Ahmedabad": (2, 0.05, "West"), "Jaipur": (2, 0.04, "North"),
    "Lucknow": (2, 0.04, "North"), "Kochi": (2, 0.03, "South"), "Indore": (2, 0.03, "West"),
    "Bhubaneswar": (2, 0.03, "East"), "Coimbatore": (2, 0.03, "South"), "Nagpur": (2, 0.02, "West"),
    "Patna": (3, 0.03, "East"), "Guwahati": (3, 0.02, "East"), "Vijayawada": (3, 0.02, "South"),
    "Ranchi": (3, 0.02, "East"), "Dehradun": (3, 0.02, "North"), "Madurai": (3, 0.02, "South"),
    "Raipur": (3, 0.01, "East"),
}

CHANNELS = {"Organic Search": 0.24, "Paid Social": 0.26, "Google Ads": 0.18,
            "Referral": 0.10, "Marketplace": 0.15, "Influencer": 0.07}
# channel -> multiplier on repeat-purchase propensity (loyalty)
CHANNEL_LOYALTY = {"Organic Search": 1.15, "Paid Social": 0.80, "Google Ads": 0.95,
                   "Referral": 1.35, "Marketplace": 0.70, "Influencer": 0.85}

CATEGORIES = {
    # category: (n_skus, price_low, price_high, margin, base_return_rate, weight)
    "Electronics Accessories": (22, 399, 4999, 0.28, 0.07, 0.20),
    "Fashion": (30, 299, 2999, 0.52, 0.18, 0.24),
    "Home & Kitchen": (24, 249, 3499, 0.40, 0.06, 0.18),
    "Beauty & Personal Care": (20, 149, 1499, 0.58, 0.04, 0.16),
    "Sports & Fitness": (14, 499, 5999, 0.35, 0.08, 0.10),
    "Books & Stationery": (12, 99, 899, 0.30, 0.02, 0.12),
}

PAYMENTS = {"UPI": 0.46, "COD": 0.22, "Credit/Debit Card": 0.18, "Wallet": 0.08, "Net Banking": 0.06}


# Demand multiplier by calendar month (festive Oct/Nov, Jan sale, monsoon dip)
SEASON = {1: 1.15, 2: 0.90, 3: 0.95, 4: 0.92, 5: 0.95, 6: 0.88, 7: 0.85,
          8: 1.00, 9: 1.05, 10: 1.45, 11: 1.35, 12: 1.10}
SEASON_MAX = max(SEASON.values())


def build_products(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for cat, (n, lo, hi, margin, ret, _w) in CATEGORIES.items():
        prices = np.exp(rng.uniform(np.log(lo), np.log(hi), n)).round(-1) - 1
        for i, p in enumerate(prices):
            m = np.clip(rng.normal(margin, 0.05), 0.1, 0.75)
            rows.append({
                "product_id": f"{cat[:3].upper()}-{i + 1:03d}",
                "category": cat,
                "list_price": float(p),
                "unit_cost": round(float(p) * (1 - m), 2),
                "base_return_rate": ret,
            })
    return pd.DataFrame(rows)


def build_customers(rng: np.random.Generator) -> pd.DataFrame:
    cities = list(CITIES)
    w = np.array([CITIES[c][1] for c in cities]); w /= w.sum()
    ch = list(CHANNELS); cw = np.array(list(CHANNELS.values())); cw /= cw.sum()

    # acquisition grows over time (brand scaling up) -> sample with increasing density.
    # Signups start in 2022 so the 2023 export window opens with an existing customer base.
    days = (END - PRE_START).days
    u = rng.uniform(0, 1, N_CUSTOMERS) ** 0.8
    signup = PRE_START + pd.to_timedelta((u * days).astype(int), unit="D")

    df = pd.DataFrame({
        "customer_id": [f"C{100000 + i}" for i in range(N_CUSTOMERS)],
        "signup_date": signup,
        "city": rng.choice(cities, N_CUSTOMERS, p=w),
        "acquisition_channel": rng.choice(ch, N_CUSTOMERS, p=cw),
        "age": np.clip(rng.normal(31, 8, N_CUSTOMERS), 18, 65).astype(int),
        "gender": rng.choice(["F", "M", "Other"], N_CUSTOMERS, p=[0.52, 0.46, 0.02]),
    })
    df["city_tier"] = df["city"].map(lambda c: CITIES[c][0])
    df["region"] = df["city"].map(lambda c: CITIES[c][2])
    return df


def build_orders(rng: np.random.Generator, customers: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    cat_names = list(CATEGORIES)
    cat_w = np.array([CATEGORIES[c][5] for c in cat_names]); cat_w /= cat_w.sum()
    pay = list(PAYMENTS); pw = np.array(list(PAYMENTS.values()))
    prods_by_cat = {c: list(g.itertuples(index=False)) for c, g in products.groupby("category")}

    records = []
    order_no = 500000
    for cust in customers.itertuples(index=False):
        loyalty = CHANNEL_LOYALTY[cust.acquisition_channel] * (1.1 if cust.city_tier == 1 else 1.0)
        # latent churn: each customer has a 'lifetime' after which they stop buying
        lifetime_days = rng.exponential(210 * loyalty)
        horizon_end = min(END, cust.signup_date + pd.Timedelta(days=lifetime_days))
        # first order shortly after signup
        t = cust.signup_date + pd.Timedelta(days=int(rng.exponential(4)))
        # mean gap between orders
        gap_mean = rng.uniform(40, 120) / loyalty
        # customer-level payment preference (tier 3 leans COD)
        p_adj = pw.copy()
        if cust.city_tier == 3:
            p_adj[pay.index("COD")] *= 2.0
        p_adj /= p_adj.sum()
        fav_cat = rng.choice(cat_names, p=cat_w)

        n = 0
        while t <= horizon_end and t <= END:
            # seasonality by thinning: festive months accept more candidate purchase moments
            if n > 0 and rng.random() > SEASON[t.month] / SEASON_MAX:
                t = t + pd.Timedelta(days=max(1, int(rng.exponential(gap_mean / SEASON_MAX))))
                continue
            if t < START:  # purchase happened before the export window -> not in the data
                n += 1
                t = t + pd.Timedelta(days=max(1, int(rng.exponential(gap_mean / SEASON_MAX))))
                continue
            cat = fav_cat if rng.random() < 0.55 else rng.choice(cat_names, p=cat_w)
            items = rng.choice([1, 1, 1, 2, 2, 3])
            payment = rng.choice(pay, p=p_adj)
            festive = t.month in (10, 11) or (t.month == 1 and t.day <= 20)
            discount = rng.choice([0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40],
                                  p=[0.25, 0.15, 0.20, 0.15, 0.12, 0.09, 0.04] if not festive
                                  else [0.05, 0.08, 0.17, 0.22, 0.22, 0.17, 0.09])
            order_no += 1
            for _ in range(items):
                pool = prods_by_cat[cat]
                pr = pool[int(rng.integers(len(pool)))]
                qty = int(rng.choice([1, 1, 1, 1, 2, 2, 3]))
                # return probability: category base + COD + deep discount + tier-3 delivery delay
                delivery_days = int(np.clip(rng.normal({1: 2.5, 2: 4, 3: 6}[cust.city_tier], 1.3), 1, 14))
                p_ret = (pr.base_return_rate
                         + (0.07 if payment == "COD" else 0)
                         + (0.04 if discount >= 0.30 else 0)
                         + 0.008 * max(delivery_days - 4, 0))
                returned = rng.random() < p_ret
                records.append({
                    "order_id": f"ORD{order_no}",
                    "order_date": t + pd.Timedelta(minutes=int(rng.integers(0, 1440))),
                    "customer_id": cust.customer_id,
                    "product_id": pr.product_id,
                    "quantity": qty,
                    "unit_price": pr.list_price,
                    "discount_pct": discount,
                    "payment_method": payment,
                    "delivery_days": delivery_days,
                    "is_returned": int(returned),
                    "rating": (int(np.clip(round(rng.normal(4.2 - 0.25 * max(delivery_days - 4, 0)
                                                             - (1.5 if returned else 0), 0.8)), 1, 5))
                               if rng.random() < 0.6 else np.nan),
                })
            n += 1
            t = t + pd.Timedelta(days=max(1, int(rng.exponential(gap_mean / SEASON_MAX))))
    return pd.DataFrame(records)


def inject_dirt(rng: np.random.Generator, orders: pd.DataFrame, customers: pd.DataFrame):
    o = orders.copy()
    n = len(o)

    # 1) mixed date formats (export from two systems)
    iso = o["order_date"].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(dtype=object)
    dmy = o["order_date"].dt.strftime("%d/%m/%Y %H:%M").to_numpy(dtype=object)
    mask = rng.random(n) < 0.18
    o["order_date"] = np.where(mask, dmy, iso)

    # 2) price stored as currency string for a slice of rows
    plain = o["unit_price"].map(lambda v: f"{v:.2f}").to_numpy(dtype=object)
    rupee = o["unit_price"].map(lambda v: f"₹{v:,.2f}").to_numpy(dtype=object)
    mask = rng.random(n) < 0.12
    o["unit_price"] = np.where(mask, rupee, plain)

    # 3) negative / zero quantity typos and absurd outliers
    idx = rng.choice(n, int(n * 0.004), replace=False); o.loc[idx, "quantity"] *= -1
    idx = rng.choice(n, int(n * 0.002), replace=False); o.loc[idx, "quantity"] = 0
    idx = rng.choice(n, 25, replace=False); o.loc[idx, "quantity"] = rng.integers(150, 999, 25)

    # 4) payment method spelling noise + missing
    noise = {"UPI": ["upi", "U.P.I", "UPI "], "COD": ["Cash on Delivery", "cod"],
             "Credit/Debit Card": ["Card", "credit card"], "Wallet": ["wallet", "Paytm Wallet"]}
    for k, variants in noise.items():
        m = (o["payment_method"] == k) & (rng.random(n) < 0.06)
        o.loc[m, "payment_method"] = rng.choice(variants, m.sum())
    o.loc[rng.random(n) < 0.01, "payment_method"] = np.nan

    # 5) duplicated rows (double-ingested batch)
    dup = o.sample(frac=0.015, random_state=SEED)
    o = pd.concat([o, dup], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

    c = customers.copy()
    c["signup_date"] = c["signup_date"].dt.strftime("%Y-%m-%d")
    variants = {"Bengaluru": ["Bangalore", "bengaluru ", "BLR"], "Mumbai": ["Bombay", "mumbai"],
                "Delhi": ["New Delhi", "delhi NCR"], "Kolkata": ["Calcutta"], "Chennai": ["Madras"],
                "Gurugram": []}
    for k, vs in variants.items():
        if not vs:
            continue
        m = (c["city"] == k) & (rng.random(len(c)) < 0.15)
        c.loc[m, "city"] = rng.choice(vs, m.sum())
    c.loc[rng.random(len(c)) < 0.02, "city"] = np.nan
    c.loc[rng.random(len(c)) < 0.015, "age"] = np.nan
    c = c.drop(columns=["city_tier", "region"])  # must be re-derived in cleaning
    return o, c


def main() -> None:
    rng = np.random.default_rng(SEED)
    RAW.mkdir(parents=True, exist_ok=True)
    products = build_products(rng)
    customers = build_customers(rng)
    orders = build_orders(rng, customers, products)
    orders_dirty, customers_dirty = inject_dirt(rng, orders, customers)

    products.drop(columns=["base_return_rate"]).to_csv(RAW / "products.csv", index=False)
    customers_dirty.to_csv(RAW / "customers_raw.csv", index=False)
    orders_dirty.to_csv(RAW / "orders_raw.csv", index=False)
    print(f"products={len(products):,} customers={len(customers_dirty):,} "
          f"order_lines_raw={len(orders_dirty):,} -> {RAW}")


if __name__ == "__main__":
    main()
