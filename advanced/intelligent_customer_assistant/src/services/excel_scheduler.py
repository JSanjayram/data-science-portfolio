"""
Real-time Excel export scheduler for EDA analysis
"""

import os
import schedule
import time
import threading
from datetime import datetime
import logging
from src.utils.database import ExcelExportService

logger = logging.getLogger(__name__)

class ExcelScheduler:
    """Scheduler for automatic Excel exports"""
    
    def __init__(self, export_interval_minutes: int = 30):
        self.export_interval = export_interval_minutes
        self.is_running = False
        self.scheduler_thread = None
        self.export_service = ExcelExportService()
        
    def start_scheduler(self):
        """Start the Excel export scheduler"""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
            
        # Schedule exports
        schedule.every(self.export_interval).minutes.do(self._export_realtime_data)
        schedule.every().hour.do(self._export_analytics_data)
        
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info(f"Excel scheduler started - exports every {self.export_interval} minutes")
        
        # Run initial export
        self._export_realtime_data()
        
    def stop_scheduler(self):
        """Stop the Excel export scheduler"""
        self.is_running = False
        schedule.clear()
        logger.info("Excel scheduler stopped")
        
    def _run_scheduler(self):
        """Run the scheduler in a separate thread"""
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
    def _export_realtime_data(self):
        """Export real-time data to Excel"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"realtime_data_{timestamp}.xlsx"
            export_path = os.path.join("data", "exports", filename)
            
            # Ensure export directory exists
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            
            exported_file = self.export_service.export_realtime_data(export_path)
            logger.info(f"Real-time data exported to: {exported_file}")
            
            # Keep only last 10 files to save space
            self._cleanup_old_exports("realtime_data_", 10)
            
        except Exception as e:
            logger.error(f"Failed to export real-time data: {str(e)}")
            
    def _export_analytics_data(self):
        """Export analytics data for EDA"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analytics_data_{timestamp}.xlsx"
            export_path = os.path.join("data", "exports", filename)
            
            exported_file = self.export_service.export_analytics_data(export_path)
            logger.info(f"Analytics data exported to: {exported_file}")
            
            # Keep only last 5 analytics files
            self._cleanup_old_exports("analytics_data_", 5)
            
        except Exception as e:
            logger.error(f"Failed to export analytics data: {str(e)}")
            
    def _cleanup_old_exports(self, prefix: str, keep_count: int):
        """Clean up old export files"""
        try:
            export_dir = os.path.join("data", "exports")
            if not os.path.exists(export_dir):
                return
                
            # Get all files with the prefix
            files = [f for f in os.listdir(export_dir) if f.startswith(prefix)]
            files.sort(reverse=True)  # Sort by name (newest first)
            
            # Remove old files
            for file_to_remove in files[keep_count:]:
                file_path = os.path.join(export_dir, file_to_remove)
                os.remove(file_path)
                logger.info(f"Removed old export file: {file_to_remove}")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old exports: {str(e)}")
    
    def export_now(self, export_type: str = "both"):
        """Manually trigger export"""
        try:
            if export_type in ["realtime", "both"]:
                self._export_realtime_data()
                
            if export_type in ["analytics", "both"]:
                self._export_analytics_data()
                
            return True
        except Exception as e:
            logger.error(f"Manual export failed: {str(e)}")
            return False

# Global scheduler instance
excel_scheduler = ExcelScheduler()