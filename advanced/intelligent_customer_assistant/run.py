#!/usr/bin/env python3
"""
Main entry point for Intelligent Customer Assistant
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from api.routes import create_app
from utils.logger import setup_logger
from utils.config import config
from services.sqlite_excel_scheduler import sqlite_excel_scheduler

def main():
    """Main application entry point"""
    logger = setup_logger(__name__)
    
    try:
        # Create Flask app
        app, assistant_service = create_app()
        
        # Get configuration
        app_config = config.get('app')
        host = app_config.get('host', '0.0.0.0')
        port = app_config.get('port', 5000)
        debug = app_config.get('debug', False)
        
        logger.info(f"Starting Intelligent Customer Assistant on {host}:{port}")
        logger.info(f"Debug mode: {debug}")
        
        # Start Excel scheduler for real-time exports
        sqlite_excel_scheduler.start_scheduler()
        
        # Run the application
        app.run(host=host, port=port, debug=debug)
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()