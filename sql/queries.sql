-- ============================================================
-- AI BUSINESS INTELLIGENCE AGENT
-- Business Intelligence SQL Queries
-- ============================================================


-- ------------------------------------------------------------
-- 1. Total Revenue
-- ------------------------------------------------------------

SELECT
    ROUND(
        SUM(
            quantity * unit_price
            - discount_amount
        ),
        2
    ) AS total_revenue
FROM orders
WHERE order_status = 'completed';


-- ------------------------------------------------------------
-- 2. Monthly Revenue
-- ------------------------------------------------------------

SELECT
    strftime('%Y-%m', order_date) AS month,
    ROUND(
        SUM(
            quantity * unit_price
            - discount_amount
        ),
        2
    ) AS revenue
FROM orders
WHERE order_status = 'completed'
GROUP BY month
ORDER BY month;


-- ------------------------------------------------------------
-- 3. Revenue by Product
-- ------------------------------------------------------------

SELECT
    p.product_id,
    p.product_name,
    p.category,
    ROUND(
        SUM(
            o.quantity * o.unit_price
            - o.discount_amount
        ),
        2
    ) AS revenue
FROM orders o
JOIN products p
    ON o.product_id = p.product_id
WHERE o.order_status = 'completed'
GROUP BY
    p.product_id,
    p.product_name,
    p.category
ORDER BY revenue DESC;


-- ------------------------------------------------------------
-- 4. Top 5 Products
-- ------------------------------------------------------------

SELECT
    p.product_name,
    p.category,
    ROUND(
        SUM(
            o.quantity * o.unit_price
            - o.discount_amount
        ),
        2
    ) AS revenue
FROM orders o
JOIN products p
    ON o.product_id = p.product_id
WHERE o.order_status = 'completed'
GROUP BY
    p.product_id,
    p.product_name,
    p.category
ORDER BY revenue DESC
LIMIT 5;


-- ------------------------------------------------------------
-- 5. Revenue by Customer Segment
-- ------------------------------------------------------------

SELECT
    s.segment_name,
    ROUND(
        SUM(
            o.quantity * o.unit_price
            - o.discount_amount
        ),
        2
    ) AS revenue
FROM orders o
JOIN customers c
    ON o.customer_id = c.customer_id
JOIN segments s
    ON c.segment_id = s.segment_id
WHERE o.order_status = 'completed'
GROUP BY
    s.segment_id,
    s.segment_name
ORDER BY revenue DESC;


-- ------------------------------------------------------------
-- 6. Orders by Country
-- ------------------------------------------------------------

SELECT
    c.country,
    COUNT(DISTINCT o.order_id) AS total_orders,
    ROUND(
        SUM(
            o.quantity * o.unit_price
            - o.discount_amount
        ),
        2
    ) AS revenue
FROM orders o
JOIN customers c
    ON o.customer_id = c.customer_id
WHERE o.order_status = 'completed'
GROUP BY c.country
ORDER BY revenue DESC;


-- ------------------------------------------------------------
-- 7. Monthly Revenue Growth
-- ------------------------------------------------------------

WITH monthly_revenue AS (
    SELECT
        strftime('%Y-%m', order_date) AS month,
        SUM(
            quantity * unit_price
            - discount_amount
        ) AS revenue
    FROM orders
    WHERE order_status = 'completed'
    GROUP BY month
)

SELECT
    month,
    ROUND(revenue, 2) AS revenue,
    ROUND(
        (
            revenue
            - LAG(revenue) OVER (
                ORDER BY month
            )
        )
        * 100.0
        / NULLIF(
            LAG(revenue) OVER (
                ORDER BY month
            ),
            0
        ),
        2
    ) AS growth_percentage
FROM monthly_revenue
ORDER BY month;


-- ------------------------------------------------------------
-- 8. Customer Revenue
-- ------------------------------------------------------------

SELECT
    c.customer_id,
    c.customer_name,
    s.segment_name,
    COUNT(o.order_id) AS order_count,
    ROUND(
        SUM(
            o.quantity * o.unit_price
            - o.discount_amount
        ),
        2
    ) AS revenue
FROM customers c
JOIN segments s
    ON c.segment_id = s.segment_id
LEFT JOIN orders o
    ON c.customer_id = o.customer_id
    AND o.order_status = 'completed'
GROUP BY
    c.customer_id,
    c.customer_name,
    s.segment_name
ORDER BY revenue DESC;


-- ------------------------------------------------------------
-- 9. Product Performance
-- ------------------------------------------------------------

SELECT
    p.product_name,
    p.category,
    COUNT(o.order_id) AS order_count,
    SUM(o.quantity) AS units_sold,
    ROUND(
        SUM(
            o.quantity * o.unit_price
            - o.discount_amount
        ),
        2
    ) AS revenue,
    ROUND(
        SUM(
            o.quantity * (
                o.unit_price - p.cost_price
            )
            - o.discount_amount
        ),
        2
    ) AS estimated_profit
FROM products p
JOIN orders o
    ON p.product_id = o.product_id
WHERE o.order_status = 'completed'
GROUP BY
    p.product_id,
    p.product_name,
    p.category
ORDER BY revenue DESC;


-- ------------------------------------------------------------
-- 10. Order Status Distribution
-- ------------------------------------------------------------

SELECT
    order_status,
    COUNT(*) AS order_count
FROM orders
GROUP BY order_status
ORDER BY order_count DESC;