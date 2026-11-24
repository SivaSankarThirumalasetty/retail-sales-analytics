-- Business questions answered in SQL (SQLite dialect).
-- Run with: python src/run_sql.py   (loads data/processed into an in-memory star schema)
-- Tables: fact_order_lines, dim_customers, dim_products

-- name: monthly_revenue_mom
-- Q1. Monthly realised revenue with month-over-month growth (window function)
WITH m AS (
    SELECT strftime('%Y-%m', order_date) AS month,
           ROUND(SUM(realised_revenue), 0) AS revenue,
           COUNT(DISTINCT order_id)        AS orders
    FROM fact_order_lines
    GROUP BY 1
)
SELECT month, revenue, orders,
       ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY month))
             / LAG(revenue) OVER (ORDER BY month), 1) AS mom_growth_pct
FROM m
ORDER BY month;

-- name: category_profitability
-- Q2. Category profitability and return rate
SELECT category,
       ROUND(SUM(realised_revenue), 0)                                 AS revenue,
       ROUND(100.0 * SUM(realised_profit) / SUM(realised_revenue), 1)  AS margin_pct,
       ROUND(100.0 * AVG(is_returned), 1)                              AS return_rate_pct
FROM fact_order_lines
GROUP BY category
ORDER BY revenue DESC;

-- name: top_cities
-- Q3. Top 10 cities by revenue with each city's share of its region (window partition)
WITH c AS (
    SELECT region, city, SUM(realised_revenue) AS revenue
    FROM fact_order_lines
    WHERE city <> 'Unknown'
    GROUP BY region, city
)
SELECT region, city, ROUND(revenue, 0) AS revenue,
       ROUND(100.0 * revenue / SUM(revenue) OVER (PARTITION BY region), 1) AS share_of_region_pct,
       RANK() OVER (ORDER BY revenue DESC) AS overall_rank
FROM c
ORDER BY revenue DESC
LIMIT 10;

-- name: cod_return_gap
-- Q4. COD vs prepaid return rate by city tier
SELECT city_tier,
       ROUND(100.0 * AVG(CASE WHEN payment_method = 'COD' THEN is_returned END), 1)  AS cod_return_pct,
       ROUND(100.0 * AVG(CASE WHEN payment_method <> 'COD' THEN is_returned END), 1) AS prepaid_return_pct,
       ROUND(100.0 * AVG(payment_method = 'COD'), 1)                                AS cod_share_pct
FROM fact_order_lines
WHERE city_tier IS NOT NULL
GROUP BY city_tier
ORDER BY city_tier;

-- name: channel_ltv
-- Q5. 12-month customer value by acquisition channel (customers acquired up to Jun 2024)
WITH first_order AS (
    SELECT customer_id, MIN(order_date) AS first_date
    FROM fact_order_lines GROUP BY customer_id
)
SELECT c.acquisition_channel,
       COUNT(DISTINCT f.customer_id)                                         AS customers,
       ROUND(SUM(f.realised_revenue) / COUNT(DISTINCT f.customer_id), 0)     AS revenue_12m_per_customer,
       ROUND(1.0 * COUNT(DISTINCT f.order_id) / COUNT(DISTINCT f.customer_id), 2) AS orders_12m_per_customer
FROM fact_order_lines f
JOIN first_order fo ON fo.customer_id = f.customer_id
JOIN dim_customers c ON c.customer_id = f.customer_id
WHERE fo.first_date < '2024-07-01'
  AND c.signup_date >= '2023-01-01'
  AND julianday(f.order_date) - julianday(fo.first_date) <= 365
GROUP BY c.acquisition_channel
ORDER BY revenue_12m_per_customer DESC;

-- name: discount_band_profit
-- Q6. Is discounting profitable? Profit per order line by discount band
SELECT discount_band,
       COUNT(*)                                   AS lines,
       ROUND(AVG(realised_profit), 0)             AS profit_per_line,
       ROUND(100.0 * AVG(is_returned), 1)         AS return_rate_pct
FROM fact_order_lines
GROUP BY discount_band
ORDER BY MIN(discount_pct);

-- name: festive_uplift
-- Q7. Festive season (Oct–Nov) uplift vs the rest of the year
SELECT strftime('%Y', order_date) AS year,
       ROUND(SUM(CASE WHEN strftime('%m', order_date) IN ('10','11') THEN realised_revenue END) / 2, 0) AS festive_month_avg,
       ROUND(SUM(CASE WHEN strftime('%m', order_date) NOT IN ('10','11') THEN realised_revenue END)
             / COUNT(DISTINCT CASE WHEN strftime('%m', order_date) NOT IN ('10','11')
                                   THEN strftime('%m', order_date) END), 0)                  AS other_month_avg
FROM fact_order_lines
GROUP BY 1
HAVING COUNT(DISTINCT strftime('%m', order_date)) = 12;
