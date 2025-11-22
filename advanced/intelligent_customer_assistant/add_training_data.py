#!/usr/bin/env python3
"""
Add comprehensive training data to improve model accuracy
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.sqlite_database import sqlite_manager
import logging

logger = logging.getLogger(__name__)

def add_comprehensive_training_data():
    """Add extensive training data for better accuracy"""
    
    training_data = [
        # Greeting variations
        ('hello', 'greeting'), ('hi', 'greeting'), ('hey', 'greeting'), ('good morning', 'greeting'),
        ('good afternoon', 'greeting'), ('good evening', 'greeting'), ('howdy', 'greeting'),
        ('greetings', 'greeting'), ('what\'s up', 'greeting'), ('hey there', 'greeting'),
        
        # Goodbye variations
        ('bye', 'goodbye'), ('goodbye', 'goodbye'), ('see you later', 'goodbye'),
        ('farewell', 'goodbye'), ('have a good day', 'goodbye'), ('thanks bye', 'goodbye'),
        ('catch you later', 'goodbye'), ('take care', 'goodbye'),
        
        # Order status variations
        ('where is my order', 'order_status'), ('track my order', 'order_status'),
        ('order status', 'order_status'), ('check my order', 'order_status'),
        ('when will my order arrive', 'order_status'), ('delivery status', 'order_status'),
        ('shipment status', 'order_status'), ('order tracking', 'order_status'),
        ('has my order shipped', 'order_status'), ('order update', 'order_status'),
        
        # Order cancellation
        ('cancel my order', 'order_cancel'), ('I want to cancel my order', 'order_cancel'),
        ('stop my order', 'order_cancel'), ('order cancellation', 'order_cancel'),
        ('cancel order', 'order_cancel'), ('I need to cancel', 'order_cancel'),
        
        # Return requests
        ('return item', 'return_request'), ('I want to return', 'return_request'),
        ('refund', 'return_request'), ('exchange product', 'return_request'),
        ('send back', 'return_request'), ('product return', 'return_request'),
        ('money back', 'return_request'), ('return policy', 'return_request'),
        ('how to return', 'return_request'), ('return this', 'return_request'),
        
        # Product inquiries
        ('product information', 'product_inquiry'), ('item details', 'product_inquiry'),
        ('product specs', 'product_inquiry'), ('tell me about', 'product_inquiry'),
        ('product features', 'product_inquiry'), ('product description', 'product_inquiry'),
        ('what is this product', 'product_inquiry'), ('product details', 'product_inquiry'),
        
        # Product search
        ('find product', 'product_search'), ('search for', 'product_search'),
        ('looking for', 'product_search'), ('do you have', 'product_search'),
        ('show me products', 'product_search'), ('browse products', 'product_search'),
        ('product catalog', 'product_search'), ('find items', 'product_search'),
        
        # Price inquiries
        ('how much', 'price_inquiry'), ('price', 'price_inquiry'), ('cost', 'price_inquiry'),
        ('pricing', 'price_inquiry'), ('expensive', 'price_inquiry'), ('cheap', 'price_inquiry'),
        ('what does it cost', 'price_inquiry'), ('how much does this cost', 'price_inquiry'),
        
        # Shipping info
        ('shipping cost', 'shipping_info'), ('delivery time', 'shipping_info'),
        ('shipping options', 'shipping_info'), ('how long', 'shipping_info'),
        ('when will it arrive', 'shipping_info'), ('delivery fee', 'shipping_info'),
        ('shipping methods', 'shipping_info'), ('express delivery', 'shipping_info'),
        
        # Payment help
        ('payment failed', 'payment_help'), ('billing issue', 'payment_help'),
        ('payment methods', 'payment_help'), ('credit card', 'payment_help'),
        ('paypal', 'payment_help'), ('payment problem', 'payment_help'),
        ('transaction failed', 'payment_help'), ('payment declined', 'payment_help'),
        
        # Account help
        ('account problem', 'account_help'), ('login issue', 'account_help'),
        ('password reset', 'account_help'), ('forgot password', 'account_help'),
        ('account locked', 'account_help'), ('profile update', 'account_help'),
        ('change email', 'account_help'), ('can\'t login', 'account_help'),
        
        # Complaints
        ('complaint', 'complaint'), ('problem', 'complaint'), ('issue', 'complaint'),
        ('not satisfied', 'complaint'), ('disappointed', 'complaint'), ('angry', 'complaint'),
        ('frustrated', 'complaint'), ('poor service', 'complaint'), ('bad experience', 'complaint'),
        
        # Compliments
        ('great service', 'compliment'), ('excellent', 'compliment'), ('amazing', 'compliment'),
        ('love it', 'compliment'), ('fantastic', 'compliment'), ('wonderful', 'compliment'),
        ('good job', 'compliment'), ('thank you', 'compliment'), ('satisfied', 'compliment'),
        
        # Technical support
        ('not working', 'technical_support'), ('broken', 'technical_support'),
        ('technical issue', 'technical_support'), ('bug', 'technical_support'),
        ('error', 'technical_support'), ('website problem', 'technical_support'),
        ('app not working', 'technical_support'), ('can\'t access', 'technical_support'),
        
        # Store hours
        ('store hours', 'store_hours'), ('opening hours', 'store_hours'),
        ('when open', 'store_hours'), ('business hours', 'store_hours'),
        ('what time', 'store_hours'), ('operating hours', 'store_hours'),
        
        # Contact info
        ('contact number', 'contact_info'), ('phone number', 'contact_info'),
        ('email', 'contact_info'), ('address', 'contact_info'),
        ('how to contact', 'contact_info'), ('customer service number', 'contact_info'),
        
        # Stock inquiry
        ('in stock', 'stock_inquiry'), ('out of stock', 'stock_inquiry'),
        ('available', 'stock_inquiry'), ('when back in stock', 'stock_inquiry'),
        ('restock', 'stock_inquiry'), ('inventory', 'stock_inquiry'),
        
        # Promotions
        ('discount', 'promotion_inquiry'), ('coupon', 'promotion_inquiry'),
        ('promo code', 'promotion_inquiry'), ('sale', 'promotion_inquiry'),
        ('offer', 'promotion_inquiry'), ('deal', 'promotion_inquiry'),
        ('special offer', 'promotion_inquiry'), ('any deals', 'promotion_inquiry'),
    ]
    
    # Clear existing training data
    with sqlite_manager.get_connection() as conn:
        conn.execute("DELETE FROM training_data")
        conn.commit()
    
    # Insert new training data
    with sqlite_manager.get_connection() as conn:
        for text, intent in training_data:
            conn.execute(
                "INSERT INTO training_data (text, intent, confidence) VALUES (?, ?, ?)",
                (text, intent, 0.95)
            )
        conn.commit()
    
    print(f"Added {len(training_data)} training examples")
    return len(training_data)

if __name__ == "__main__":
    count = add_comprehensive_training_data()
    print(f"Successfully added {count} training examples to the database")
    print("Now run: python train_model.py")