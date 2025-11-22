"""
Database connection and management utilities
"""

import os
import psycopg2
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database connections for company and e-commerce databases"""
    
    def __init__(self):
        self.company_engine = None
        self.ecommerce_engine = None
        self._initialize_connections()
    
    def _initialize_connections(self):
        """Initialize database connections"""
        try:
            # Company database connection
            company_url = os.getenv('COMPANY_DB_URL', 'postgresql://assistant_user:assistant_pass@localhost:5432/company_db')
            self.company_engine = create_engine(
                company_url,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True
            )
            
            # E-commerce database connection
            ecommerce_url = os.getenv('ECOMMERCE_DB_URL', 'postgresql://assistant_user:assistant_pass@localhost:5432/ecommerce_db')
            self.ecommerce_engine = create_engine(
                ecommerce_url,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True
            )
            
            logger.info("Database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database connections: {str(e)}")
            raise
    
    @contextmanager
    def get_company_connection(self):
        """Get company database connection"""
        conn = self.company_engine.connect()
        try:
            yield conn
        finally:
            conn.close()
    
    @contextmanager
    def get_ecommerce_connection(self):
        """Get e-commerce database connection"""
        conn = self.ecommerce_engine.connect()
        try:
            yield conn
        finally:
            conn.close()
    
    def execute_query(self, query: str, params: Dict = None, database: str = 'ecommerce') -> List[Dict]:
        """Execute query and return results"""
        engine = self.ecommerce_engine if database == 'ecommerce' else self.company_engine
        
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                if result.returns_rows:
                    columns = result.keys()
                    return [dict(zip(columns, row)) for row in result.fetchall()]
                return []
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise
    
    def get_dataframe(self, query: str, params: Dict = None, database: str = 'ecommerce') -> pd.DataFrame:
        """Execute query and return pandas DataFrame"""
        engine = self.ecommerce_engine if database == 'ecommerce' else self.company_engine
        
        try:
            return pd.read_sql(query, engine, params=params)
        except Exception as e:
            logger.error(f"DataFrame query failed: {str(e)}")
            raise
    
    def export_to_excel(self, queries: Dict[str, str], filename: str, database: str = 'ecommerce'):
        """Export multiple queries to Excel with different sheets"""
        engine = self.ecommerce_engine if database == 'ecommerce' else self.company_engine
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                for sheet_name, query in queries.items():
                    df = pd.read_sql(query, engine)
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            logger.info(f"Data exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Excel export failed: {str(e)}")
            raise

# Global database manager instance
db_manager = DatabaseManager()

class CompanyDataService:
    """Service for company knowledge base operations"""
    
    @staticmethod
    def get_company_info(category: str = None) -> List[Dict]:
        """Get company information by category"""
        query = "SELECT * FROM company_info"
        params = {}
        
        if category:
            query += " WHERE category = :category"
            params['category'] = category
            
        return db_manager.execute_query(query, params, 'company')
    
    @staticmethod
    def get_intents() -> List[Dict]:
        """Get all intents"""
        query = "SELECT * FROM intents ORDER BY intent_name"
        return db_manager.execute_query(query, database='company')
    
    @staticmethod
    def get_training_data() -> pd.DataFrame:
        """Get training data for model"""
        query = "SELECT text, intent, entities FROM training_data"
        return db_manager.get_dataframe(query, database='company')
    
    @staticmethod
    def add_training_data(text: str, intent: str, entities: Dict = None):
        """Add new training data"""
        query = """
        INSERT INTO training_data (text, intent, entities) 
        VALUES (:text, :intent, :entities)
        """
        params = {
            'text': text,
            'intent': intent,
            'entities': entities
        }
        db_manager.execute_query(query, params, 'company')

class EcommerceDataService:
    """Service for e-commerce data operations"""
    
    @staticmethod
    def get_customer_info(customer_id: str) -> Dict:
        """Get customer information"""
        query = "SELECT * FROM customers WHERE customer_id = :customer_id"
        result = db_manager.execute_query(query, {'customer_id': customer_id})
        return result[0] if result else None
    
    @staticmethod
    def get_order_status(order_id: str) -> Dict:
        """Get order status and details"""
        query = """
        SELECT o.*, s.status as shipment_status, s.tracking_number, s.estimated_delivery
        FROM orders o
        LEFT JOIN shipments s ON o.order_id = s.order_id
        WHERE o.order_id = :order_id
        """
        result = db_manager.execute_query(query, {'order_id': order_id})
        return result[0] if result else None
    
    @staticmethod
    def get_customer_orders(customer_id: str) -> List[Dict]:
        """Get all orders for a customer"""
        query = """
        SELECT o.*, COUNT(oi.id) as item_count
        FROM orders o
        LEFT JOIN order_items oi ON o.order_id = oi.order_id
        WHERE o.customer_id = :customer_id
        GROUP BY o.id
        ORDER BY o.order_date DESC
        """
        return db_manager.execute_query(query, {'customer_id': customer_id})
    
    @staticmethod
    def search_products(search_term: str, category_id: int = None) -> List[Dict]:
        """Search products by name or description"""
        query = """
        SELECT p.*, c.name as category_name
        FROM products p
        LEFT JOIN categories c ON p.category_id = c.id
        WHERE p.status = 'active'
        AND (p.name ILIKE :search_term OR p.description ILIKE :search_term)
        """
        params = {'search_term': f'%{search_term}%'}
        
        if category_id:
            query += " AND p.category_id = :category_id"
            params['category_id'] = category_id
            
        return db_manager.execute_query(query, params)
    
    @staticmethod
    def get_product_details(product_id: str) -> Dict:
        """Get detailed product information"""
        query = """
        SELECT p.*, c.name as category_name,
               AVG(pr.rating) as avg_rating,
               COUNT(pr.id) as review_count
        FROM products p
        LEFT JOIN categories c ON p.category_id = c.id
        LEFT JOIN product_reviews pr ON p.product_id = pr.product_id
        WHERE p.product_id = :product_id
        GROUP BY p.id, c.name
        """
        result = db_manager.execute_query(query, {'product_id': product_id})
        return result[0] if result else None
    
    @staticmethod
    def log_interaction(customer_id: str, session_id: str, message: str, 
                       intent: str, confidence: float, response: str, response_time: int):
        """Log customer interaction"""
        query = """
        INSERT INTO customer_interactions 
        (customer_id, session_id, interaction_type, message, intent, confidence, response, response_time_ms)
        VALUES (:customer_id, :session_id, 'chat', :message, :intent, :confidence, :response, :response_time)
        """
        params = {
            'customer_id': customer_id,
            'session_id': session_id,
            'message': message,
            'intent': intent,
            'confidence': confidence,
            'response': response,
            'response_time': response_time
        }
        db_manager.execute_query(query, params)

class ExcelExportService:
    """Service for real-time Excel exports"""
    
    @staticmethod
    def export_realtime_data(export_path: str = None) -> str:
        """Export real-time data to Excel"""
        if not export_path:
            export_path = os.path.join('data', 'exports', f'realtime_data_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
        
        queries = {
            'customers': "SELECT * FROM customers ORDER BY registration_date DESC",
            'orders': """
                SELECT o.*, c.first_name, c.last_name, c.email
                FROM orders o
                JOIN customers c ON o.customer_id = c.customer_id
                ORDER BY o.order_date DESC
            """,
            'products': """
                SELECT p.*, c.name as category_name, 
                       COALESCE(AVG(pr.rating), 0) as avg_rating,
                       COUNT(pr.id) as review_count
                FROM products p
                LEFT JOIN categories c ON p.category_id = c.id
                LEFT JOIN product_reviews pr ON p.product_id = pr.product_id
                GROUP BY p.id, c.name
                ORDER BY p.created_at DESC
            """,
            'shipments': """
                SELECT s.*, o.customer_id, sm.name as shipping_method
                FROM shipments s
                JOIN orders o ON s.order_id = o.order_id
                JOIN shipping_methods sm ON s.shipping_method_id = sm.id
                ORDER BY s.created_at DESC
            """,
            'interactions': """
                SELECT ci.*, c.first_name, c.last_name
                FROM customer_interactions ci
                LEFT JOIN customers c ON ci.customer_id = c.customer_id
                ORDER BY ci.created_at DESC
                LIMIT 1000
            """,
            'support_tickets': """
                SELECT st.*, c.first_name, c.last_name, c.email
                FROM support_tickets st
                LEFT JOIN customers c ON st.customer_id = c.customer_id
                ORDER BY st.created_at DESC
            """,
            'sales_summary': """
                SELECT 
                    DATE(order_date) as order_date,
                    COUNT(*) as total_orders,
                    SUM(total_amount) as total_revenue,
                    AVG(total_amount) as avg_order_value
                FROM orders
                WHERE order_date >= CURRENT_DATE - INTERVAL '30 days'
                GROUP BY DATE(order_date)
                ORDER BY order_date DESC
            """
        }
        
        return db_manager.export_to_excel(queries, export_path)
    
    @staticmethod
    def export_analytics_data(export_path: str = None) -> str:
        """Export analytics data for EDA"""
        if not export_path:
            export_path = os.path.join('data', 'exports', f'analytics_data_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
        
        queries = {
            'customer_metrics': """
                SELECT 
                    c.customer_id,
                    c.first_name,
                    c.last_name,
                    c.registration_date,
                    COUNT(o.id) as total_orders,
                    COALESCE(SUM(o.total_amount), 0) as total_spent,
                    COALESCE(AVG(o.total_amount), 0) as avg_order_value,
                    MAX(o.order_date) as last_order_date
                FROM customers c
                LEFT JOIN orders o ON c.customer_id = o.customer_id
                GROUP BY c.customer_id, c.first_name, c.last_name, c.registration_date
            """,
            'product_performance': """
                SELECT 
                    p.product_id,
                    p.name,
                    p.price,
                    p.stock_quantity,
                    COUNT(oi.id) as times_ordered,
                    COALESCE(SUM(oi.quantity), 0) as total_quantity_sold,
                    COALESCE(SUM(oi.total_price), 0) as total_revenue,
                    COALESCE(AVG(pr.rating), 0) as avg_rating
                FROM products p
                LEFT JOIN order_items oi ON p.product_id = oi.product_id
                LEFT JOIN product_reviews pr ON p.product_id = pr.product_id
                GROUP BY p.product_id, p.name, p.price, p.stock_quantity
            """,
            'monthly_trends': """
                SELECT 
                    DATE_TRUNC('month', order_date) as month,
                    COUNT(*) as total_orders,
                    SUM(total_amount) as total_revenue,
                    COUNT(DISTINCT customer_id) as unique_customers
                FROM orders
                WHERE order_date >= CURRENT_DATE - INTERVAL '12 months'
                GROUP BY DATE_TRUNC('month', order_date)
                ORDER BY month
            """,
            'intent_analysis': """
                SELECT 
                    intent,
                    COUNT(*) as frequency,
                    AVG(confidence) as avg_confidence,
                    AVG(response_time_ms) as avg_response_time
                FROM customer_interactions
                WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
                GROUP BY intent
                ORDER BY frequency DESC
            """
        }
        
        return db_manager.export_to_excel(queries, export_path, 'ecommerce')