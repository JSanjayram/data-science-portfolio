-- Connect to company_db and insert sample data
\c company_db;

-- Insert company information
INSERT INTO company_info (category, title, content, tags) VALUES
('about', 'Company Overview', 'We are a leading e-commerce platform offering quality products with excellent customer service.', ARRAY['company', 'about']),
('policy', 'Return Policy', 'Items can be returned within 30 days of purchase with original packaging.', ARRAY['return', 'policy']),
('policy', 'Shipping Policy', 'Free shipping on orders over $50. Standard delivery takes 3-5 business days.', ARRAY['shipping', 'policy']),
('support', 'Contact Information', 'Customer support available 24/7 via chat, email, or phone.', ARRAY['contact', 'support']),
('faq', 'Payment Methods', 'We accept all major credit cards, PayPal, and digital wallets.', ARRAY['payment', 'faq']);

-- Insert intents
INSERT INTO intents (intent_name, description, examples, response_templates) VALUES
('greeting', 'User greets the assistant', ARRAY['hello', 'hi', 'good morning', 'hey'], ARRAY['Hello! How can I help you today?', 'Hi there! What can I do for you?']),
('order_status', 'User asks about order status', ARRAY['where is my order', 'order status', 'track my order'], ARRAY['Let me check your order status for you.', 'I can help you track your order.']),
('return_request', 'User wants to return an item', ARRAY['return item', 'refund', 'exchange product'], ARRAY['I can help you with your return request.', 'Let me assist you with the return process.']),
('product_inquiry', 'User asks about products', ARRAY['product information', 'item details', 'product specs'], ARRAY['What product would you like to know about?', 'I can provide product information for you.']),
('shipping_info', 'User asks about shipping', ARRAY['shipping cost', 'delivery time', 'shipping options'], ARRAY['Let me provide you with shipping information.', 'Here are our shipping options.']),
('payment_help', 'User needs payment assistance', ARRAY['payment failed', 'billing issue', 'payment methods'], ARRAY['I can help you with payment issues.', 'Let me assist with your payment concern.']),
('complaint', 'User has a complaint', ARRAY['complaint', 'problem', 'issue', 'not satisfied'], ARRAY['I understand your concern. Let me help resolve this.', 'I apologize for the inconvenience. How can I help?']);

-- Insert entities
INSERT INTO entities (entity_name, entity_type, synonyms, patterns) VALUES
('order_number', 'identifier', ARRAY['order id', 'order number', 'reference number'], ARRAY['[A-Z]{2}[0-9]{6}', '[0-9]{8,10}']),
('product_name', 'product', ARRAY['item', 'product', 'article'], ARRAY[]),
('email', 'contact', ARRAY['email address', 'e-mail'], ARRAY['[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}']),
('phone', 'contact', ARRAY['phone number', 'mobile', 'telephone'], ARRAY['[0-9]{10}', '\+[0-9]{1,3}[0-9]{10}']);

-- Connect to ecommerce_db and insert sample data
\c ecommerce_db;

-- Insert categories
INSERT INTO categories (name, description) VALUES
('Electronics', 'Electronic devices and accessories'),
('Clothing', 'Fashion and apparel'),
('Home & Garden', 'Home improvement and garden supplies'),
('Books', 'Books and educational materials'),
('Sports', 'Sports and fitness equipment');

-- Insert products
INSERT INTO products (product_id, name, description, category_id, price, cost_price, stock_quantity, sku, brand, weight, dimensions, images, tags, status) VALUES
('ELEC001', 'Wireless Bluetooth Headphones', 'High-quality wireless headphones with noise cancellation', 1, 99.99, 45.00, 150, 'WBH-001', 'TechSound', 0.35, '{"length": 18, "width": 16, "height": 8}', ARRAY['headphones1.jpg', 'headphones2.jpg'], ARRAY['wireless', 'bluetooth', 'audio'], 'active'),
('ELEC002', 'Smartphone Case', 'Protective case for smartphones', 1, 24.99, 8.50, 300, 'SPC-002', 'ProtectPro', 0.15, '{"length": 15, "width": 8, "height": 1}', ARRAY['case1.jpg'], ARRAY['phone', 'protection'], 'active'),
('CLOTH001', 'Cotton T-Shirt', 'Comfortable cotton t-shirt', 2, 19.99, 7.50, 200, 'CTS-001', 'ComfortWear', 0.20, '{"size": "M"}', ARRAY['tshirt1.jpg', 'tshirt2.jpg'], ARRAY['cotton', 'casual'], 'active'),
('HOME001', 'LED Desk Lamp', 'Adjustable LED desk lamp', 3, 45.99, 18.00, 75, 'LDL-001', 'BrightLight', 1.20, '{"length": 40, "width": 20, "height": 50}', ARRAY['lamp1.jpg'], ARRAY['led', 'desk', 'lighting'], 'active'),
('BOOK001', 'Programming Guide', 'Complete programming guide for beginners', 4, 34.99, 12.00, 100, 'PG-001', 'TechBooks', 0.80, '{"pages": 450}', ARRAY['book1.jpg'], ARRAY['programming', 'education'], 'active');

-- Insert customers
INSERT INTO customers (customer_id, first_name, last_name, email, phone, address, registration_date, status) VALUES
('CUST001', 'John', 'Doe', 'john.doe@email.com', '+1234567890', '{"street": "123 Main St", "city": "New York", "state": "NY", "zip": "10001", "country": "USA"}', '2024-01-15', 'active'),
('CUST002', 'Jane', 'Smith', 'jane.smith@email.com', '+1234567891', '{"street": "456 Oak Ave", "city": "Los Angeles", "state": "CA", "zip": "90210", "country": "USA"}', '2024-02-20', 'active'),
('CUST003', 'Mike', 'Johnson', 'mike.johnson@email.com', '+1234567892', '{"street": "789 Pine Rd", "city": "Chicago", "state": "IL", "zip": "60601", "country": "USA"}', '2024-03-10', 'active');

-- Insert shipping methods
INSERT INTO shipping_methods (name, description, base_cost, estimated_days, is_active) VALUES
('Standard Shipping', 'Regular delivery service', 5.99, 5, true),
('Express Shipping', 'Fast delivery service', 12.99, 2, true),
('Overnight Shipping', 'Next day delivery', 24.99, 1, true),
('Free Shipping', 'Free standard shipping for orders over $50', 0.00, 7, true);

-- Insert orders
INSERT INTO orders (order_id, customer_id, order_date, status, total_amount, tax_amount, shipping_amount, payment_method, payment_status, shipping_address, billing_address) VALUES
('ORD001', 'CUST001', '2024-10-01 10:30:00', 'delivered', 124.98, 10.00, 5.99, 'credit_card', 'completed', '{"street": "123 Main St", "city": "New York", "state": "NY", "zip": "10001"}', '{"street": "123 Main St", "city": "New York", "state": "NY", "zip": "10001"}'),
('ORD002', 'CUST002', '2024-10-05 14:15:00', 'shipped', 64.98, 5.20, 12.99, 'paypal', 'completed', '{"street": "456 Oak Ave", "city": "Los Angeles", "state": "CA", "zip": "90210"}', '{"street": "456 Oak Ave", "city": "Los Angeles", "state": "CA", "zip": "90210"}'),
('ORD003', 'CUST003', '2024-10-08 09:45:00', 'processing', 45.99, 3.68, 0.00, 'credit_card', 'completed', '{"street": "789 Pine Rd", "city": "Chicago", "state": "IL", "zip": "60601"}', '{"street": "789 Pine Rd", "city": "Chicago", "state": "IL", "zip": "60601"}');

-- Insert order items
INSERT INTO order_items (order_id, product_id, quantity, unit_price, total_price) VALUES
('ORD001', 'ELEC001', 1, 99.99, 99.99),
('ORD001', 'ELEC002', 1, 24.99, 24.99),
('ORD002', 'CLOTH001', 2, 19.99, 39.98),
('ORD002', 'ELEC002', 1, 24.99, 24.99),
('ORD003', 'HOME001', 1, 45.99, 45.99);

-- Insert shipments
INSERT INTO shipments (shipment_id, order_id, shipping_method_id, tracking_number, carrier, status, shipped_date, estimated_delivery, delivery_address) VALUES
('SHIP001', 'ORD001', 1, 'TRK123456789', 'FedEx', 'delivered', '2024-10-02 08:00:00', '2024-10-07 17:00:00', '{"street": "123 Main St", "city": "New York", "state": "NY", "zip": "10001"}'),
('SHIP002', 'ORD002', 2, 'TRK987654321', 'UPS', 'in_transit', '2024-10-06 10:30:00', '2024-10-08 16:00:00', '{"street": "456 Oak Ave", "city": "Los Angeles", "state": "CA", "zip": "90210"}');

-- Insert support tickets
INSERT INTO support_tickets (ticket_id, customer_id, subject, description, status, priority, category) VALUES
('TKT001', 'CUST001', 'Product not working', 'The headphones I received are not connecting to my device', 'open', 'high', 'product_issue'),
('TKT002', 'CUST002', 'Shipping delay', 'My order was supposed to arrive yesterday but has not been delivered', 'in_progress', 'medium', 'shipping');

-- Insert product reviews
INSERT INTO product_reviews (product_id, customer_id, rating, title, review_text, is_verified_purchase) VALUES
('ELEC001', 'CUST001', 4, 'Good quality headphones', 'Sound quality is excellent, but the battery life could be better', true),
('CLOTH001', 'CUST002', 5, 'Perfect fit', 'Very comfortable and good quality fabric', true);

-- Insert sample customer interactions
INSERT INTO customer_interactions (customer_id, session_id, interaction_type, message, intent, confidence, response, response_time_ms) VALUES
('CUST001', 'sess_001', 'chat', 'Hello, I need help with my order', 'greeting', 0.95, 'Hello! I can help you with your order. What do you need assistance with?', 150),
('CUST002', 'sess_002', 'chat', 'Where is my order ORD002?', 'order_status', 0.89, 'Let me check the status of order ORD002 for you.', 200),
('CUST003', 'sess_003', 'chat', 'I want to return this item', 'return_request', 0.92, 'I can help you with your return request. Which item would you like to return?', 180);