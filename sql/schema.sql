-- ============================================================
-- AI BUSINESS INTELLIGENCE AGENT
-- Business Database Schema
-- ============================================================

PRAGMA foreign_keys = ON;

-- ------------------------------------------------------------
-- Segments
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS segments (
    segment_id INTEGER PRIMARY KEY,
    segment_name TEXT NOT NULL UNIQUE,
    description TEXT
);

-- ------------------------------------------------------------
-- Customers
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY,
    customer_name TEXT NOT NULL,
    email TEXT UNIQUE,
    segment_id INTEGER,
    signup_date DATE,
    country TEXT,
    acquisition_channel TEXT,
    FOREIGN KEY (segment_id)
        REFERENCES segments(segment_id)
);

-- ------------------------------------------------------------
-- Products
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT NOT NULL,
    category TEXT NOT NULL,
    unit_price REAL NOT NULL,
    cost_price REAL NOT NULL
);

-- ------------------------------------------------------------
-- Orders
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    order_date DATE NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    discount_amount REAL DEFAULT 0,
    order_status TEXT NOT NULL,

    FOREIGN KEY (customer_id)
        REFERENCES customers(customer_id),

    FOREIGN KEY (product_id)
        REFERENCES products(product_id)
);

-- ------------------------------------------------------------
-- Business Metrics
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS business_metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_date DATE NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    metric_category TEXT
);

-- ------------------------------------------------------------
-- Indexes
-- ------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_orders_customer
ON orders(customer_id);

CREATE INDEX IF NOT EXISTS idx_orders_product
ON orders(product_id);

CREATE INDEX IF NOT EXISTS idx_orders_date
ON orders(order_date);

CREATE INDEX IF NOT EXISTS idx_customers_segment
ON customers(segment_id);

CREATE INDEX IF NOT EXISTS idx_metrics_date
ON business_metrics(metric_date);