"""
SQLite-based Excel export scheduler
"""

import os
import schedule
import time
import threading
from datetime import datetime
import logging
from src.utils.sqlite_database import SQLiteDataService

logger = logging.getLogger(__name__)

class SQLiteExcelScheduler:
    """SQLite-based Excel scheduler"""
    
    def __init__(self, export_interval_minutes: int = 30):
        self.export_interval = export_interval_minutes
        self.is_running = False
        self.scheduler_thread = None
        
    def start_scheduler(self):
        """Start the Excel export scheduler"""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
            
        schedule.every(self.export_interval).minutes.do(self._export_realtime_data)
        
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info(f"SQLite Excel scheduler started - exports every {self.export_interval} minutes")
        
        # Run initial export
        self._export_realtime_data()
        
    def stop_scheduler(self):
        """Stop scheduler"""
        self.is_running = False
        schedule.clear()
        logger.info("SQLite Excel scheduler stopped")
        
    def _run_scheduler(self):
        """Run scheduler in thread"""
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)
            
    def _export_realtime_data(self):
        """Export real-time data"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"realtime_data_{timestamp}.xlsx"
            export_path = os.path.join("data", "exports", filename)
            
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            
            exported_file = SQLiteDataService.export_realtime_data(export_path)
            logger.info(f"Real-time data exported to: {exported_file}")
            
            self._cleanup_old_exports("realtime_data_", 10)
            
        except Exception as e:
            logger.error(f"Failed to export real-time data: {str(e)}")
            
    def _cleanup_old_exports(self, prefix: str, keep_count: int):
        """Clean up old files"""
        try:
            export_dir = os.path.join("data", "exports")
            if not os.path.exists(export_dir):
                return
                
            files = [f for f in os.listdir(export_dir) if f.startswith(prefix)]
            files.sort(reverse=True)
            
            for file_to_remove in files[keep_count:]:
                file_path = os.path.join(export_dir, file_to_remove)
                os.remove(file_path)
                logger.info(f"Removed old export file: {file_to_remove}")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old exports: {str(e)}")
    
    def export_now(self, export_type: str = "realtime"):
        """Manual export trigger"""
        try:
            self._export_realtime_data()
            return True
        except Exception as e:
            logger.error(f"Manual export failed: {str(e)}")
            return False

# Global scheduler
sqlite_excel_scheduler = SQLiteExcelScheduler()