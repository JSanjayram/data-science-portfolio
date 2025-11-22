from typing import Dict, List
import time
from collections import defaultdict

class MetricsCollector:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.counters = defaultdict(int)
    
    def record_response_time(self, endpoint: str, response_time: float):
        self.metrics[f'{endpoint}_response_time'].append(response_time)
    
    def increment_counter(self, metric_name: str):
        self.counters[metric_name] += 1
    
    def record_confidence(self, confidence: float):
        self.metrics['confidence_scores'].append(confidence)
    
    def get_avg_response_time(self, endpoint: str) -> float:
        times = self.metrics.get(f'{endpoint}_response_time', [])
        return sum(times) / len(times) if times else 0
    
    def get_avg_confidence(self) -> float:
        scores = self.metrics.get('confidence_scores', [])
        return sum(scores) / len(scores) if scores else 0
    
    def get_metrics_summary(self) -> Dict:
        return {
            'total_requests': self.counters.get('requests', 0),
            'avg_confidence': self.get_avg_confidence(),
            'avg_response_time': self.get_avg_response_time('chat'),
            'error_count': self.counters.get('errors', 0)
        }