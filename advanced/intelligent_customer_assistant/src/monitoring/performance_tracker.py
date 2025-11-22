import time
from functools import wraps
from typing import Callable, Dict
from .metrics import MetricsCollector

class PerformanceTracker:
    def __init__(self):
        self.metrics = MetricsCollector()
    
    def track_time(self, operation_name: str):
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    self.metrics.increment_counter(f'{operation_name}_success')
                    return result
                except Exception as e:
                    self.metrics.increment_counter(f'{operation_name}_error')
                    raise
                finally:
                    end_time = time.time()
                    self.metrics.record_response_time(operation_name, end_time - start_time)
            return wrapper
        return decorator
    
    def get_performance_stats(self) -> Dict:
        return self.metrics.get_metrics_summary()
    
    def reset_metrics(self):
        self.metrics = MetricsCollector()