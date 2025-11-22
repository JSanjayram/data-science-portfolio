-- Sample Data for Sales Analytics System
USE sales_analytics;

-- Insert sample customers
INSERT INTO customers (customer_name, email, phone, city, state, country, customer_segment, total_lifetime_value) VALUES
('Tech Solutions Inc', 'contact@techsolutions.com', '+1-555-0101', 'New York', 'NY', 'USA', 'VIP', 125000.00),
('Global Enterprises', 'info@globalent.com', '+1-555-0102', 'Los Angeles', 'CA', 'USA', 'Premium', 85000.00),
('StartUp Innovations', 'hello@startup.com', '+1-555-0103', 'Austin', 'TX', 'USA', 'Standard', 35000.00),
('Retail Chain Co', 'orders@retailchain.com', '+1-555-0104', 'Chicago', 'IL', 'USA', 'Premium', 95000.00),
('Small Business LLC', 'owner@smallbiz.com', '+1-555-0105', 'Miami', 'FL', 'USA', 'Basic', 15000.00),
('Manufacturing Corp', 'procurement@mfgcorp.com', '+1-555-0106', 'Detroit', 'MI', 'USA', 'VIP', 150000.00),
('Digital Agency', 'team@digitalagency.com', '+1-555-0107', 'Seattle', 'WA', 'USA', 'Standard', 45000.00),
('Healthcare Systems', 'it@healthcare.com', '+1-555-0108', 'Boston', 'MA', 'USA', 'Premium', 75000.00);

-- Insert sample products
INSERT INTO products (product_code, product_name, product_line, category, price, msrp, cost, stock_quantity) VALUES
('LAPTOP-001', 'Business Laptop Pro', 'Computers', 'Hardware', 1299.99, 1499.99, 899.99, 50),
('SOFT-001', 'CRM Software License', 'Software', 'Business Software', 299.99, 399.99, 150.00, 100),
('SERV-001', 'IT Support Package', 'Services', 'Support', 199.99, 249.99, 100.00, 25),
('PHONE-001', 'Business Smartphone', 'Mobile', 'Hardware', 799.99, 899.99, 599.99, 75),
('TABLET-001', 'Professional Tablet', 'Tablets', 'Hardware', 599.99, 699.99, 449.99, 40),
('SOFT-002', 'Analytics Platform', 'Software', 'Analytics', 499.99, 599.99, 250.00, 30),
('SERV-002', 'Cloud Hosting', 'Services', 'Infrastructure', 99.99, 129.99, 50.00, 200),
('ACC-001', 'Wireless Accessories Kit', 'Accessories', 'Hardware', 149.99, 199.99, 89.99, 120);

-- Insert sample orders
INSERT INTO orders (order_number, customer_id, order_date, status, total_amount, territory, sales_rep) VALUES
('ORD-2024-001', 1, '2024-01-15', 'Delivered', 2599.98, 'North America', 'John Smith'),
('ORD-2024-002', 2, '2024-01-18', 'Shipped', 1799.97, 'North America', 'Sarah Johnson'),
('ORD-2024-003', 3, '2024-01-20', 'Processing', 899.98, 'North America', 'Mike Wilson'),
('ORD-2024-004', 4, '2024-01-22', 'Delivered', 3299.96, 'North America', 'Lisa Brown'),
('ORD-2024-005', 5, '2024-01-25', 'Pending', 449.99, 'North America', 'David Lee'),
('ORD-2024-006', 6, '2024-01-28', 'Shipped', 4599.95, 'North America', 'John Smith'),
('ORD-2024-007', 7, '2024-02-01', 'Delivered', 1299.98, 'North America', 'Sarah Johnson'),
('ORD-2024-008', 8, '2024-02-03', 'Processing', 2199.97, 'North America', 'Mike Wilson');

-- Insert sample order details
INSERT INTO order_details (order_id, product_id, quantity_ordered, price_each, sales_amount, profit_margin) VALUES
(1, 1, 2, 1299.99, 2599.98, 30.77),
(2, 4, 1, 799.99, 799.99, 25.00),
(2, 2, 1, 299.99, 299.99, 50.00),
(2, 8, 1, 149.99, 149.99, 40.00),
(3, 5, 1, 599.99, 599.99, 25.00),
(3, 2, 1, 299.99, 299.99, 50.00),
(4, 1, 1, 1299.99, 1299.99, 30.77),
(4, 6, 2, 499.99, 999.98, 50.00),
(4, 3, 1, 199.99, 199.99, 50.00),
(5, 8, 3, 149.99, 449.97, 40.00),
(6, 1, 2, 1299.99, 2599.98, 30.77),
(6, 6, 2, 499.99, 999.98, 50.00),
(6, 3, 5, 199.99, 999.95, 50.00),
(7, 1, 1, 1299.99, 1299.99, 30.77),
(8, 4, 2, 799.99, 1599.98, 25.00),
(8, 5, 1, 599.99, 599.99, 25.00);

-- Insert realistic customer interactions for AI training
INSERT INTO customer_interactions (customer_id, order_id, interaction_type, customer_query, intent_category, ai_response, resolution_status, satisfaction_score) VALUES
(1, 1, 'order_status', 'Hi, I need to check the status of my order ORD-2024-001. When will it arrive?', 'order_tracking', 'Your order ORD-2024-001 has been delivered on January 20th. You should have received a delivery confirmation email. If you haven''t received it, please check with your reception or neighbors.', 'resolved', 5),

(2, 2, 'billing', 'I was charged $1799.97 but I think there might be an error in my invoice. Can you help?', 'billing_inquiry', 'I can help you with your billing inquiry. Your order total of $1799.97 includes: Business Smartphone ($799.99), CRM Software License ($299.99), and Wireless Accessories Kit ($149.99), plus applicable taxes. Would you like me to email you a detailed breakdown?', 'resolved', 4),

(3, 3, 'product_info', 'What are the technical specifications of the Professional Tablet I ordered?', 'product_specifications', 'The Professional Tablet (TABLET-001) features: 12.9-inch display, 256GB storage, 8GB RAM, all-day battery life, and includes stylus support. It''s perfect for business presentations and mobile productivity. Would you like more detailed specs?', 'resolved', 5),

(4, 4, 'complaint', 'My Analytics Platform software is not working properly after installation. I keep getting error messages.', 'technical_support', 'I''m sorry to hear about the technical issues with your Analytics Platform. Let me connect you with our technical support team who can help troubleshoot the installation. They''ll contact you within 2 hours with a solution.', 'escalated', 3),

(5, 5, 'order_status', 'My order is still showing as pending. It''s been 3 days since I placed it. What''s the delay?', 'order_delay', 'I apologize for the delay with order ORD-2024-005. Due to high demand for the Wireless Accessories Kit, there''s a 2-day processing delay. Your order will ship tomorrow and you''ll receive tracking information via email.', 'resolved', 4),

(6, 6, 'inquiry', 'I need to add more IT Support Packages to my existing order. Is that possible?', 'order_modification', 'I can help you add more IT Support Packages. Since your order ORD-2024-006 is currently shipped, I''ll create a new order for the additional packages. You''ll get the same corporate discount. How many additional packages do you need?', 'resolved', 5),

(7, 7, 'support', 'The laptop I received has a small scratch on the screen. Can I get a replacement?', 'product_defect', 'I''m sorry about the scratched screen on your Business Laptop Pro. This is covered under our quality guarantee. I''ll arrange for a replacement laptop to be shipped today, and you can return the damaged one using the prepaid label I''ll email you.', 'resolved', 5),

(8, 8, 'billing', 'Can I change my payment method for the current order? I want to use a different credit card.', 'payment_modification', 'Since your order ORD-2024-008 is currently processing, I can update the payment method. Please provide the new card details securely through our payment portal link I''ll send you. The order will continue processing once payment is confirmed.', 'resolved', 4),

(1, NULL, 'product_info', 'Do you have any new laptop models coming out this quarter? I might need to upgrade our fleet.', 'product_inquiry', 'Great question! We have two new business laptop models launching in March: the Business Laptop Pro Max with enhanced graphics and the Business Laptop Lite for basic productivity. I can arrange a demo for your team. Would you like me to schedule that?', 'resolved', 5),

(2, NULL, 'support', 'I need training for my team on the CRM Software we purchased. Do you offer training sessions?', 'training_request', 'Absolutely! We offer comprehensive CRM training sessions. For Premium customers like yourself, we provide free online training sessions and can arrange on-site training for larger teams. I''ll have our training coordinator contact you to schedule a session.', 'resolved', 5),

(3, NULL, 'inquiry', 'What''s your return policy for software licenses if we don''t end up using them?', 'return_policy', 'Our software licenses have a 30-day return policy for unused licenses. Since software licenses are digital products, we can deactivate them and process a full refund within the return window. Physical products have a 30-day return policy as well.', 'resolved', 4),

(4, NULL, 'complaint', 'The Cloud Hosting service has been experiencing downtime. This is affecting our business operations.', 'service_complaint', 'I sincerely apologize for the Cloud Hosting downtime affecting your operations. This is unacceptable for a Premium customer. I''m escalating this to our infrastructure team immediately and will ensure you receive service credits for the downtime. You''ll hear from our technical director within the hour.', 'escalated', 2),

(5, NULL, 'billing', 'I received a bill but I don''t remember placing an order. Can you check what this charge is for?', 'billing_dispute', 'I can help you identify this charge. Looking at your account, the recent bill is for order ORD-2024-005 placed on January 25th for Wireless Accessories Kit ($449.97). I can email you the original order confirmation and invoice details for your records.', 'resolved', 4),

(6, NULL, 'product_info', 'We''re expanding our operations. Can you provide bulk pricing for 50+ Business Laptops?', 'bulk_pricing', 'Excellent! For VIP customers like Manufacturing Corp, we offer significant volume discounts. For 50+ Business Laptop Pro units, you''ll get 15% off plus free setup and deployment services. I''ll have our enterprise sales team prepare a custom quote for you within 24 hours.', 'resolved', 5),

(7, NULL, 'support', 'Our team needs help integrating the Analytics Platform with our existing systems. Do you provide integration support?', 'integration_support', 'Yes, we provide full integration support for the Analytics Platform. Our technical team can help with API setup, data migration, and custom integrations. For Standard customers, this service is available at $150/hour. Would you like me to schedule a consultation call?', 'resolved', 4),

(8, NULL, 'inquiry', 'What are your payment terms for large orders? We''re planning a major IT upgrade.', 'payment_terms', 'For Premium customers like Healthcare Systems, we offer flexible payment terms including NET 30, NET 60, and installment plans for orders over $10,000. We also provide early payment discounts. I''ll have our finance team contact you to discuss the best terms for your IT upgrade project.', 'resolved', 5);