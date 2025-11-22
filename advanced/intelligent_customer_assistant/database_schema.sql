-- Sales Analytics Database Schema
-- Core tables for customer service and sales data

-- 1. CUSTOMERS TABLE
CREATE TABLE customers (
    customer_id INT PRIMARY KEY AUTO_INCREMENT,
    customer_name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE,
    phone VARCHAR(20),
    address TEXT,
    city VARCHAR(100),
    state VARCHAR(50),
    country VARCHAR(50),
    postal_code VARCHAR(20),
    registration_date DATE,
    customer_segment ENUM('VIP', 'Premium', 'Standard', 'Basic') DEFAULT 'Standard',
    total_lifetime_value DECIMAL(12,2) DEFAULT 0.00,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. PRODUCTS TABLE
CREATE TABLE products (
    product_id INT PRIMARY KEY AUTO_INCREMENT,
    product_code VARCHAR(50) UNIQUE,
    product_name VARCHAR(255) NOT NULL,
    product_line VARCHAR(100),
    category VARCHAR(100),
    price DECIMAL(10,2),
    msrp DECIMAL(10,2),
    cost DECIMAL(10,2),
    stock_quantity INT DEFAULT 0,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. ORDERS TABLE
CREATE TABLE orders (
    order_id INT PRIMARY KEY AUTO_INCREMENT,
    order_number VARCHAR(50) UNIQUE,
    customer_id INT,
    order_date DATE,
    required_date DATE,
    shipped_date DATE,
    status ENUM('Pending', 'Processing', 'Shipped', 'Delivered', 'Cancelled') DEFAULT 'Pending',
    total_amount DECIMAL(12,2),
    shipping_cost DECIMAL(8,2) DEFAULT 0.00,
    tax_amount DECIMAL(8,2) DEFAULT 0.00,
    territory VARCHAR(50),
    sales_rep VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- 4. ORDER_DETAILS TABLE
CREATE TABLE order_details (
    detail_id INT PRIMARY KEY AUTO_INCREMENT,
    order_id INT,
    product_id INT,
    quantity_ordered INT,
    price_each DECIMAL(10,2),
    order_line_number INT,
    sales_amount DECIMAL(12,2),
    profit_margin DECIMAL(5,2),
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- 5. CUSTOMER_INTERACTIONS TABLE (For AI training)
CREATE TABLE customer_interactions (
    interaction_id INT PRIMARY KEY AUTO_INCREMENT,
    customer_id INT,
    order_id INT NULL,
    interaction_type ENUM('inquiry', 'complaint', 'support', 'feedback', 'order_status', 'billing', 'product_info') NOT NULL,
    customer_query TEXT NOT NULL,
    intent_category VARCHAR(100),
    ai_response TEXT,
    resolution_status ENUM('resolved', 'pending', 'escalated') DEFAULT 'pending',
    satisfaction_score INT CHECK (satisfaction_score BETWEEN 1 AND 5),
    interaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_date TIMESTAMP NULL,
    agent_id VARCHAR(50),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);