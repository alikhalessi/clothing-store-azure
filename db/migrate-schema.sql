-- ============================
-- CREATE TABLES
-- ============================

-- Categories (just primary key)
CREATE TABLE IF NOT EXISTS categories (
    category_id SERIAL PRIMARY KEY
);

-- Products (primary key + foreign key)
CREATE TABLE IF NOT EXISTS products (
    product_id SERIAL PRIMARY KEY,
    category_id INT REFERENCES categories(category_id)
);

-- Customers (primary key)
CREATE TABLE IF NOT EXISTS customers (
    customer_id SERIAL PRIMARY KEY
);

-- Orders (primary key + foreign key)
CREATE TABLE IF NOT EXISTS orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customers(customer_id)
);

-- Order Items (primary key + foreign keys)
CREATE TABLE IF NOT EXISTS order_items (
    order_item_id SERIAL PRIMARY KEY,
    order_id INT REFERENCES orders(order_id),
    product_id INT REFERENCES products(product_id)
);

-- ============================
-- Add remaining columns
-- 
-- NOTE: If you need more columns, just add them here and run this whole file again!
-- 
-- ============================

-- Categories
ALTER TABLE categories ADD COLUMN IF NOT EXISTS name VARCHAR NOT NULL;

-- Products
ALTER TABLE products ADD COLUMN IF NOT EXISTS name VARCHAR NOT NULL;
ALTER TABLE products ADD COLUMN IF NOT EXISTS price NUMERIC NOT NULL;
ALTER TABLE products ADD COLUMN IF NOT EXISTS stock INT NOT NULL;

-- Customers
ALTER TABLE customers ADD COLUMN IF NOT EXISTS first_name VARCHAR NOT NULL;
ALTER TABLE customers ADD COLUMN IF NOT EXISTS last_name VARCHAR NOT NULL;
ALTER TABLE customers ADD COLUMN IF NOT EXISTS email VARCHAR UNIQUE NOT NULL;

-- Orders
ALTER TABLE orders ADD COLUMN IF NOT EXISTS order_date DATE DEFAULT CURRENT_DATE;

-- Order Items
ALTER TABLE order_items ADD COLUMN IF NOT EXISTS quantity INT NOT NULL;