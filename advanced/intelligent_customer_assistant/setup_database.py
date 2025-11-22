#!/usr/bin/env python3
"""
Database Setup Script
"""

import subprocess
import time
import mysql.connector
import sys

def start_database():
    """Start MySQL database using Docker Compose"""
    print("Starting MySQL database with Docker Compose...")
    
    try:
        # Start the database
        result = subprocess.run([
            'docker-compose', '-f', 'docker-compose-db.yml', 'up', '-d'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Database container started successfully!")
            print("Waiting for database to be ready...")
            time.sleep(30)  # Wait for MySQL to initialize
            return True
        else:
            print(f"Error starting database: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("Docker Compose not found. Please install Docker and Docker Compose.")
        return False

def test_connection():
    """Test database connection"""
    config = {
        'host': 'localhost',
        'port': 3306,
        'user': 'sales_user',
        'password': 'sales_password',
        'database': 'sales_analytics'
    }
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            connection = mysql.connector.connect(**config)
            cursor = connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM customers")
            result = cursor.fetchone()
            
            print(f"Database connection successful!")
            print(f"Found {result[0]} customers in database")
            
            cursor.close()
            connection.close()
            return True
            
        except mysql.connector.Error as e:
            print(f"Attempt {attempt + 1}: Connection failed - {e}")
            if attempt < max_retries - 1:
                time.sleep(10)
            else:
                print("Failed to connect to database after multiple attempts")
                return False

def main():
    """Main setup function"""
    print("=== Database Setup for Intelligent Customer Assistant ===")
    
    # Start database
    if start_database():
        # Test connection
        if test_connection():
            print("\n✅ Database setup completed successfully!")
            print("\n📊 Access phpMyAdmin at: http://localhost:8080")
            print("   Username: sales_user")
            print("   Password: sales_password")
            print("\n🚀 You can now run the improved model:")
            print("   python improved_model.py")
        else:
            print("\n❌ Database connection failed")
    else:
        print("\n❌ Database setup failed")

if __name__ == '__main__':
    main()