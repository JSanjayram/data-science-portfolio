#!/usr/bin/env python3
"""
Setup script for Intelligent Customer Assistant with databases
"""

import os
import sys
import subprocess
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(command, cwd=None):
    """Run shell command and return result"""
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Command failed: {command}")
            logger.error(f"Error: {result.stderr}")
            return False
        return True
    except Exception as e:
        logger.error(f"Failed to run command {command}: {str(e)}")
        return False

def setup_docker():
    """Setup Docker containers"""
    logger.info("Setting up Docker containers...")
    
    # Stop any existing containers
    run_command("docker-compose -f docker-compose-full.yml down")
    
    # Start PostgreSQL and Redis
    if not run_command("docker-compose -f docker-compose-full.yml up -d postgres redis"):
        logger.error("Failed to start Docker containers")
        return False
    
    # Wait for PostgreSQL to be ready
    logger.info("Waiting for PostgreSQL to be ready...")
    time.sleep(30)
    
    return True

def install_dependencies():
    """Install Python dependencies"""
    logger.info("Installing Python dependencies...")
    return run_command("pip install -r requirements.txt")

def create_directories():
    """Create necessary directories"""
    directories = [
        "data/exports",
        "data/logs",
        "data/models",
        "data/processed",
        "data/raw"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def test_database_connection():
    """Test database connection"""
    logger.info("Testing database connection...")
    
    try:
        import psycopg2
        
        # Test company database
        conn = psycopg2.connect(
            host="localhost",
            port="5432",
            database="company_db",
            user="assistant_user",
            password="assistant_pass"
        )
        conn.close()
        logger.info("Company database connection successful")
        
        # Test e-commerce database
        conn = psycopg2.connect(
            host="localhost",
            port="5432",
            database="ecommerce_db",
            user="assistant_user",
            password="assistant_pass"
        )
        conn.close()
        logger.info("E-commerce database connection successful")
        
        return True
        
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        return False

def main():
    """Main setup function"""
    logger.info("Starting Intelligent Customer Assistant setup...")
    
    # Create directories
    create_directories()
    
    # Setup Docker
    if not setup_docker():
        logger.error("Docker setup failed")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        logger.error("Dependency installation failed")
        sys.exit(1)
    
    # Test database connection
    if not test_database_connection():
        logger.error("Database connection test failed")
        sys.exit(1)
    
    logger.info("Setup completed successfully!")
    logger.info("You can now run the application with: python run.py")
    logger.info("Access the web interface at: http://localhost:5000")
    logger.info("Excel exports will be saved to: data/exports/")

if __name__ == "__main__":
    main()