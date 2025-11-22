"""
SQLite database implementation for development without Docker
"""

import sqlite3
import pandas as pd
import os
import json
from typing import Dict, List, Any, Optional
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class SQLiteManager:
    """SQLite database manager for development"""
    
    def __init__(self):
        self.db_path = os.path.join('data', 'assistant.db')
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database with schema"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Company tables
            conn.executescript("""
                -- Company Knowledge Base
                CREATE TABLE IF NOT EXISTS company_info (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tags TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS intents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    intent_name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    examples TEXT,
                    response_templates TEXT,
                    confidence_threshold REAL DEFAULT 0.7,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    synonyms TEXT,
                    patterns TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS training_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    entities TEXT,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- E-commerce tables
                CREATE TABLE IF NOT EXISTS customers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id TEXT UNIQUE NOT NULL,
                    first_name TEXT,
                    last_name TEXT,
                    email TEXT UNIQUE,
                    phone TEXT,
                    address TEXT,
                    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    status TEXT DEFAULT 'active'
                );
                
                CREATE TABLE IF NOT EXISTS categories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    parent_id INTEGER REFERENCES categories(id),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS products (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    category_id INTEGER REFERENCES categories(id),
                    price REAL NOT NULL,
                    cost_price REAL,
                    stock_quantity INTEGER DEFAULT 0,
                    sku TEXT UNIQUE,
                    brand TEXT,
                    weight REAL,
                    dimensions TEXT,
                    images TEXT,
                    tags TEXT,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT UNIQUE NOT NULL,
                    customer_id TEXT REFERENCES customers(customer_id),
                    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'pending',
                    total_amount REAL NOT NULL,
                    tax_amount REAL DEFAULT 0,
                    shipping_amount REAL DEFAULT 0,
                    discount_amount REAL DEFAULT 0,
                    payment_method TEXT,
                    payment_status TEXT DEFAULT 'pending',
                    shipping_address TEXT,
                    billing_address TEXT,
                    notes TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS order_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT REFERENCES orders(order_id),
                    product_id TEXT REFERENCES products(product_id),
                    quantity INTEGER NOT NULL,
                    unit_price REAL NOT NULL,
                    total_price REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS shipping_methods (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    base_cost REAL,
                    estimated_days INTEGER,
                    is_active BOOLEAN DEFAULT 1
                );
                
                CREATE TABLE IF NOT EXISTS shipments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    shipment_id TEXT UNIQUE NOT NULL,
                    order_id TEXT REFERENCES orders(order_id),
                    shipping_method_id INTEGER REFERENCES shipping_methods(id),
                    tracking_number TEXT,
                    carrier TEXT,
                    status TEXT DEFAULT 'preparing',
                    shipped_date TIMESTAMP,
                    estimated_delivery TIMESTAMP,
                    actual_delivery TIMESTAMP,
                    delivery_address TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS support_tickets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_id TEXT UNIQUE NOT NULL,
                    customer_id TEXT REFERENCES customers(customer_id),
                    subject TEXT NOT NULL,
                    description TEXT,
                    status TEXT DEFAULT 'open',
                    priority TEXT DEFAULT 'medium',
                    category TEXT,
                    assigned_to TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS product_reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_id TEXT REFERENCES products(product_id),
                    customer_id TEXT REFERENCES customers(customer_id),
                    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
                    title TEXT,
                    review_text TEXT,
                    is_verified_purchase BOOLEAN DEFAULT 0,
                    helpful_votes INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS customer_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id TEXT,
                    session_id TEXT,
                    interaction_type TEXT,
                    message TEXT,
                    intent TEXT,
                    confidence REAL,
                    response TEXT,
                    response_time_ms INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
        self._insert_sample_data()
        logger.info("SQLite database initialized successfully")
    
    def _insert_sample_data(self):
        """Insert sample data"""
        with sqlite3.connect(self.db_path) as conn:
            # Check if data already exists
            cursor = conn.execute("SELECT COUNT(*) FROM customers")
            if cursor.fetchone()[0] > 0:
                return
            
            # Insert sample data
            conn.executescript("""
                -- Company info
                INSERT INTO company_info (category, title, content, tags) VALUES
                ('about', 'Company Overview', 'We are a leading e-commerce platform offering quality products with excellent customer service.', 'company,about'),
                ('policy', 'Return Policy', 'Items can be returned within 30 days of purchase with original packaging.', 'return,policy'),
                ('policy', 'Shipping Policy', 'Free shipping on orders over $50. Standard delivery takes 3-5 business days.', 'shipping,policy'),
                ('support', 'Contact Information', 'Customer support available 24/7 via chat, email, or phone.', 'contact,support'),
                ('faq', 'Payment Methods', 'We accept all major credit cards, PayPal, and digital wallets.', 'payment,faq');
                
                -- Comprehensive Intents
                INSERT INTO intents (intent_name, description, examples, response_templates) VALUES
                ('greeting', 'User greets the assistant', 'hello,hi,good morning,good afternoon,good evening,hey there,greetings,howdy,what\'s up', 'Hello! How can I help you today?|Hi there! What can I do for you?|Good day! I\'m here to assist you.|Welcome! How may I help you?'),
                ('goodbye', 'User says goodbye', 'bye,goodbye,see you later,thanks bye,have a good day,farewell,catch you later', 'Goodbye! Have a great day!|Thank you for contacting us. Take care!|See you later! Feel free to reach out anytime.'),
                ('order_status', 'User asks about order status', 'where is my order,order status,track my order,check order,order tracking,delivery status,shipment status,when will my order arrive', 'Let me check your order status for you.|I can help you track your order.|Please provide your order number so I can check the status.'),
                ('order_cancel', 'User wants to cancel order', 'cancel my order,cancel order,stop my order,I want to cancel,order cancellation', 'I can help you cancel your order.|Let me assist with your order cancellation.|I\'ll help you with the cancellation process.'),
                ('return_request', 'User wants to return an item', 'return item,refund,exchange product,send back,product return,money back,return policy,how to return', 'I can help you with your return request.|Let me assist you with the return process.|I\'ll guide you through our return policy.'),
                ('product_inquiry', 'User asks about products', 'product information,item details,product specs,tell me about,product features,availability,in stock,product description', 'What product would you like to know about?|I can provide product information for you.|Which item are you interested in learning about?'),
                ('product_search', 'User searches for products', 'find product,search for,looking for,do you have,show me products,browse products,product catalog', 'I can help you find products.|What are you looking for today?|Let me help you search our catalog.'),
                ('price_inquiry', 'User asks about pricing', 'how much,price,cost,pricing,expensive,cheap,discount,sale,offer,deal', 'I can help you with pricing information.|Let me check the current price for you.|Would you like to know about any current deals?'),
                ('shipping_info', 'User asks about shipping', 'shipping cost,delivery time,shipping options,how long,when will it arrive,delivery fee,shipping methods,express delivery', 'Let me provide you with shipping information.|Here are our shipping options.|I can help you with delivery details.'),
                ('payment_help', 'User needs payment assistance', 'payment failed,billing issue,payment methods,credit card,paypal,payment problem,transaction failed,payment declined', 'I can help you with payment issues.|Let me assist with your payment concern.|I\'ll help resolve your billing issue.'),
                ('account_help', 'User needs account assistance', 'account problem,login issue,password reset,forgot password,account locked,profile update,change email', 'I can help you with your account.|Let me assist with your login issue.|I\'ll help you reset your password.'),
                ('complaint', 'User has a complaint', 'complaint,problem,issue,not satisfied,disappointed,angry,frustrated,poor service,bad experience', 'I understand your concern. Let me help resolve this.|I apologize for the inconvenience. How can I help?|I\'m sorry to hear about this issue. Let me assist you.'),
                ('compliment', 'User gives positive feedback', 'great service,excellent,amazing,love it,fantastic,wonderful,good job,thank you,satisfied,happy', 'Thank you for your kind words!|I\'m glad I could help!|We appreciate your positive feedback!'),
                ('technical_support', 'User needs technical help', 'not working,broken,technical issue,bug,error,website problem,app not working,can\'t access', 'I can help you with technical issues.|Let me assist with this technical problem.|I\'ll help troubleshoot this issue.'),
                ('store_hours', 'User asks about store hours', 'store hours,opening hours,when open,business hours,what time,store timing,operating hours', 'Our customer service is available 24/7.|I\'m here to help you anytime.|We\'re always open to assist you online.'),
                ('contact_info', 'User asks for contact information', 'contact number,phone number,email,address,how to contact,customer service number,support email', 'You can reach us through this chat, email, or phone.|I can provide you with our contact information.|How would you prefer to be contacted?'),
                ('warranty_info', 'User asks about warranty', 'warranty,guarantee,product warranty,how long warranty,warranty claim,warranty period', 'I can help you with warranty information.|Let me check the warranty details for your product.|I\'ll provide warranty information.'),
                ('size_guide', 'User asks about sizing', 'size guide,what size,sizing chart,measurements,fit,size recommendation,too big,too small', 'I can help you with sizing information.|Let me provide our size guide.|I\'ll help you find the right size.'),
                ('stock_inquiry', 'User asks about stock availability', 'in stock,out of stock,available,when back in stock,restock,inventory,stock status', 'Let me check the stock status for you.|I can verify product availability.|I\'ll check our current inventory.'),
                ('promotion_inquiry', 'User asks about promotions', 'discount,coupon,promo code,sale,offer,deal,promotion,special offer,voucher', 'Let me check current promotions for you.|I can help you find available discounts.|Here are our current offers.'),
                ('gift_card', 'User asks about gift cards', 'gift card,gift certificate,voucher,gift,present,gift card balance,redeem gift card', 'I can help you with gift card information.|Let me assist with your gift card query.|I\'ll help you with gift card options.');
                
                -- Categories
                INSERT INTO categories (name, description) VALUES
                ('Electronics', 'Electronic devices and accessories'),
                ('Clothing', 'Fashion and apparel'),
                ('Home & Garden', 'Home improvement and garden supplies'),
                ('Books', 'Books and educational materials'),
                ('Sports', 'Sports and fitness equipment');
                
                -- Products
                INSERT INTO products (product_id, name, description, category_id, price, cost_price, stock_quantity, sku, brand, weight, dimensions, images, tags, status) VALUES
                ('ELEC001', 'Wireless Bluetooth Headphones', 'High-quality wireless headphones with noise cancellation', 1, 99.99, 45.00, 150, 'WBH-001', 'TechSound', 0.35, '{"length": 18, "width": 16, "height": 8}', 'headphones1.jpg,headphones2.jpg', 'wireless,bluetooth,audio', 'active'),
                ('ELEC002', 'Smartphone Case', 'Protective case for smartphones', 1, 24.99, 8.50, 300, 'SPC-002', 'ProtectPro', 0.15, '{"length": 15, "width": 8, "height": 1}', 'case1.jpg', 'phone,protection', 'active'),
                ('CLOTH001', 'Cotton T-Shirt', 'Comfortable cotton t-shirt', 2, 19.99, 7.50, 200, 'CTS-001', 'ComfortWear', 0.20, '{"size": "M"}', 'tshirt1.jpg,tshirt2.jpg', 'cotton,casual', 'active'),
                ('HOME001', 'LED Desk Lamp', 'Adjustable LED desk lamp', 3, 45.99, 18.00, 75, 'LDL-001', 'BrightLight', 1.20, '{"length": 40, "width": 20, "height": 50}', 'lamp1.jpg', 'led,desk,lighting', 'active'),
                ('BOOK001', 'Programming Guide', 'Complete programming guide for beginners', 4, 34.99, 12.00, 100, 'PG-001', 'TechBooks', 0.80, '{"pages": 450}', 'book1.jpg', 'programming,education', 'active');
                
                -- Customers
                INSERT INTO customers (customer_id, first_name, last_name, email, phone, address, registration_date, status) VALUES
                ('CUST001', 'John', 'Doe', 'john.doe@email.com', '+1234567890', '{"street": "123 Main St", "city": "New York", "state": "NY", "zip": "10001", "country": "USA"}', '2024-01-15', 'active'),
                ('CUST002', 'Jane', 'Smith', 'jane.smith@email.com', '+1234567891', '{"street": "456 Oak Ave", "city": "Los Angeles", "state": "CA", "zip": "90210", "country": "USA"}', '2024-02-20', 'active'),
                ('CUST003', 'Mike', 'Johnson', 'mike.johnson@email.com', '+1234567892', '{"street": "789 Pine Rd", "city": "Chicago", "state": "IL", "zip": "60601", "country": "USA"}', '2024-03-10', 'active');
                
                -- Shipping methods
                INSERT INTO shipping_methods (name, description, base_cost, estimated_days, is_active) VALUES
                ('Standard Shipping', 'Regular delivery service', 5.99, 5, 1),
                ('Express Shipping', 'Fast delivery service', 12.99, 2, 1),
                ('Overnight Shipping', 'Next day delivery', 24.99, 1, 1),
                ('Free Shipping', 'Free standard shipping for orders over $50', 0.00, 7, 1);
                
                -- Orders
                INSERT INTO orders (order_id, customer_id, order_date, status, total_amount, tax_amount, shipping_amount, payment_method, payment_status, shipping_address, billing_address) VALUES
                ('ORD001', 'CUST001', '2024-10-01 10:30:00', 'delivered', 124.98, 10.00, 5.99, 'credit_card', 'completed', '{"street": "123 Main St", "city": "New York", "state": "NY", "zip": "10001"}', '{"street": "123 Main St", "city": "New York", "state": "NY", "zip": "10001"}'),
                ('ORD002', 'CUST002', '2024-10-05 14:15:00', 'shipped', 64.98, 5.20, 12.99, 'paypal', 'completed', '{"street": "456 Oak Ave", "city": "Los Angeles", "state": "CA", "zip": "90210"}', '{"street": "456 Oak Ave", "city": "Los Angeles", "state": "CA", "zip": "90210"}'),
                ('ORD003', 'CUST003', '2024-10-08 09:45:00', 'processing', 45.99, 3.68, 0.00, 'credit_card', 'completed', '{"street": "789 Pine Rd", "city": "Chicago", "state": "IL", "zip": "60601"}', '{"street": "789 Pine Rd", "city": "Chicago", "state": "IL", "zip": "60601"}');
                
                -- Order items
                INSERT INTO order_items (order_id, product_id, quantity, unit_price, total_price) VALUES
                ('ORD001', 'ELEC001', 1, 99.99, 99.99),
                ('ORD001', 'ELEC002', 1, 24.99, 24.99),
                ('ORD002', 'CLOTH001', 2, 19.99, 39.98),
                ('ORD002', 'ELEC002', 1, 24.99, 24.99),
                ('ORD003', 'HOME001', 1, 45.99, 45.99);
                
                -- Shipments
                INSERT INTO shipments (shipment_id, order_id, shipping_method_id, tracking_number, carrier, status, shipped_date, estimated_delivery, delivery_address) VALUES
                ('SHIP001', 'ORD001', 1, 'TRK123456789', 'FedEx', 'delivered', '2024-10-02 08:00:00', '2024-10-07 17:00:00', '{"street": "123 Main St", "city": "New York", "state": "NY", "zip": "10001"}'),
                ('SHIP002', 'ORD002', 2, 'TRK987654321', 'UPS', 'in_transit', '2024-10-06 10:30:00', '2024-10-08 16:00:00', '{"street": "456 Oak Ave", "city": "Los Angeles", "state": "CA", "zip": "90210"}');
                
                -- Support tickets
                INSERT INTO support_tickets (ticket_id, customer_id, subject, description, status, priority, category) VALUES
                ('TKT001', 'CUST001', 'Product not working', 'The headphones I received are not connecting to my device', 'open', 'high', 'product_issue'),
                ('TKT002', 'CUST002', 'Shipping delay', 'My order was supposed to arrive yesterday but has not been delivered', 'in_progress', 'medium', 'shipping');
                
                -- Product reviews
                INSERT INTO product_reviews (product_id, customer_id, rating, title, review_text, is_verified_purchase) VALUES
                ('ELEC001', 'CUST001', 4, 'Good quality headphones', 'Sound quality is excellent, but the battery life could be better', 1),
                ('CLOTH001', 'CUST002', 5, 'Perfect fit', 'Very comfortable and good quality fabric', 1);
                
                -- Comprehensive training data
                INSERT INTO training_data (text, intent, confidence) VALUES
                ('hello', 'greeting', 0.95), ('hi there', 'greeting', 0.95), ('good morning', 'greeting', 0.95),
                ('where is my order', 'order_status', 0.92), ('track my order', 'order_status', 0.92), ('order status', 'order_status', 0.92),
                ('cancel my order', 'order_cancel', 0.90), ('I want to cancel', 'order_cancel', 0.90),
                ('return this item', 'return_request', 0.88), ('I want a refund', 'return_request', 0.88),
                ('how much does this cost', 'price_inquiry', 0.85), ('what is the price', 'price_inquiry', 0.85),
                ('find headphones', 'product_search', 0.87), ('looking for shoes', 'product_search', 0.87),
                ('shipping cost', 'shipping_info', 0.89), ('delivery time', 'shipping_info', 0.89),
                ('payment failed', 'payment_help', 0.91), ('billing issue', 'payment_help', 0.91),
                ('forgot my password', 'account_help', 0.93), ('login problem', 'account_help', 0.93),
                ('not satisfied', 'complaint', 0.86), ('poor service', 'complaint', 0.86),
                ('excellent service', 'compliment', 0.94), ('thank you', 'compliment', 0.94),
                ('website not working', 'technical_support', 0.88), ('app crashed', 'technical_support', 0.88),
                ('store hours', 'store_hours', 0.92), ('when are you open', 'store_hours', 0.92),
                ('contact number', 'contact_info', 0.90), ('customer service email', 'contact_info', 0.90),
                ('product warranty', 'warranty_info', 0.87), ('warranty period', 'warranty_info', 0.87),
                ('size guide', 'size_guide', 0.89), ('what size should I get', 'size_guide', 0.89),
                ('in stock', 'stock_inquiry', 0.91), ('available', 'stock_inquiry', 0.91),
                ('any discounts', 'promotion_inquiry', 0.88), ('promo code', 'promotion_inquiry', 0.88),
                ('gift card', 'gift_card', 0.90), ('gift certificate', 'gift_card', 0.90),
                ('goodbye', 'goodbye', 0.95), ('bye', 'goodbye', 0.95), ('see you later', 'goodbye', 0.95);
                
                -- Sample interactions
                INSERT INTO customer_interactions (customer_id, session_id, interaction_type, message, intent, confidence, response, response_time_ms) VALUES
                ('CUST001', 'sess_001', 'chat', 'Hello, I need help with my order', 'greeting', 0.95, 'Hello! I can help you with your order. What do you need assistance with?', 150),
                ('CUST002', 'sess_002', 'chat', 'Where is my order ORD002?', 'order_status', 0.89, 'Let me check the status of order ORD002 for you.', 200),
                ('CUST003', 'sess_003', 'chat', 'I want to return this item', 'return_request', 0.92, 'I can help you with your return request. Which item would you like to return?', 180);
            """)
    
    @contextmanager
    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """Execute query and return results"""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_dataframe(self, query: str, params: tuple = ()) -> pd.DataFrame:
        """Execute query and return pandas DataFrame"""
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def export_to_excel(self, queries: Dict[str, str], filename: str):
        """Export multiple queries to Excel"""
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            for sheet_name, query in queries.items():
                df = self.get_dataframe(query)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        return filename

# Global SQLite manager
sqlite_manager = SQLiteManager()

class SQLiteDataService:
    """SQLite-based data service"""
    
    @staticmethod
    def get_customer_info(customer_id: str) -> Dict:
        """Get customer information"""
        query = "SELECT * FROM customers WHERE customer_id = ?"
        result = sqlite_manager.execute_query(query, (customer_id,))
        return result[0] if result else None
    
    @staticmethod
    def get_order_status(order_id: str) -> Dict:
        """Get order status"""
        query = """
        SELECT o.*, s.status as shipment_status, s.tracking_number, s.estimated_delivery
        FROM orders o
        LEFT JOIN shipments s ON o.order_id = s.order_id
        WHERE o.order_id = ?
        """
        result = sqlite_manager.execute_query(query, (order_id,))
        return result[0] if result else None
    
    @staticmethod
    def search_products(search_term: str, category_id: int = None) -> List[Dict]:
        """Search products"""
        query = """
        SELECT p.*, c.name as category_name
        FROM products p
        LEFT JOIN categories c ON p.category_id = c.id
        WHERE p.status = 'active'
        AND (p.name LIKE ? OR p.description LIKE ?)
        """
        params = [f'%{search_term}%', f'%{search_term}%']
        
        if category_id:
            query += " AND p.category_id = ?"
            params.append(category_id)
            
        return sqlite_manager.execute_query(query, params)
    
    @staticmethod
    def log_interaction(customer_id: str, session_id: str, message: str, 
                       intent: str, confidence: float, response: str, response_time: int):
        """Log customer interaction"""
        query = """
        INSERT INTO customer_interactions 
        (customer_id, session_id, interaction_type, message, intent, confidence, response, response_time_ms)
        VALUES (?, ?, 'chat', ?, ?, ?, ?, ?)
        """
        with sqlite_manager.get_connection() as conn:
            conn.execute(query, (customer_id, session_id, message, intent, confidence, response, response_time))
            conn.commit()
    
    @staticmethod
    def export_realtime_data(export_path: str) -> str:
        """Export real-time data"""
        queries = {
            'customers': "SELECT * FROM customers ORDER BY registration_date DESC",
            'orders': """
                SELECT o.*, c.first_name, c.last_name, c.email
                FROM orders o
                JOIN customers c ON o.customer_id = c.customer_id
                ORDER BY o.order_date DESC
            """,
            'products': """
                SELECT p.*, c.name as category_name
                FROM products p
                LEFT JOIN categories c ON p.category_id = c.id
                ORDER BY p.created_at DESC
            """,
            'interactions': """
                SELECT ci.*, c.first_name, c.last_name
                FROM customer_interactions ci
                LEFT JOIN customers c ON ci.customer_id = c.customer_id
                ORDER BY ci.created_at DESC
                LIMIT 1000
            """
        }
        return sqlite_manager.export_to_excel(queries, export_path)