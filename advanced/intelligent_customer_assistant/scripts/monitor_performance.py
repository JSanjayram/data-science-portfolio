#!/usr/bin/env python3
"""
Performance monitoring script
"""

import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.monitoring.performance_tracker import PerformanceTracker
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def monitor_performance():
    """Monitor system performance"""
    tracker = PerformanceTracker()
    
    while True:
        try:
            stats = tracker.get_performance_stats()
            logger.info(f"Performance stats: {stats}")
            time.sleep(60)  # Monitor every minute
            
        except KeyboardInterrupt:
            logger.info("Monitoring stopped")
            break
        except Exception as e:
            logger.error(f"Monitoring error: {str(e)}")

if __name__ == '__main__':
    monitor_performance()