"""Performance monitoring utilities"""

import time
import functools
from typing import Callable, Any
from logger import SystemLogger

def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"⏱️  {func.__name__} executed in {execution_time:.3f}s")
        
        return result
    return wrapper

class PerformanceMonitor:
    """Monitor system performance metrics"""
    
    def __init__(self):
        self.metrics = {
            "query_times": [],
            "indexing_times": [],
            "retrieval_times": [],
            "generation_times": []
        }
    
    def record_query_time(self, duration: float) -> None:
        """Record query processing time"""
        self.metrics["query_times"].append(duration)
    
    def record_indexing_time(self, duration: float) -> None:
        """Record document indexing time"""
        self.metrics["indexing_times"].append(duration)
    
    def record_retrieval_time(self, duration: float) -> None:
        """Record document retrieval time"""
        self.metrics["retrieval_times"].append(duration)
    
    def record_generation_time(self, duration: float) -> None:
        """Record response generation time"""
        self.metrics["generation_times"].append(duration)
    
    def get_average_times(self) -> dict:
        """Get average execution times"""
        return {
            metric: sum(times) / len(times) if times else 0
            for metric, times in self.metrics.items()
        }
    
    def get_stats(self) -> dict:
        """Get detailed statistics"""
        stats = {}
        
        for metric, times in self.metrics.items():
            if times:
                stats[metric] = {
                    "avg": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "count": len(times)
                }
            else:
                stats[metric] = {
                    "avg": 0,
                    "min": 0,
                    "max": 0,
                    "count": 0
                }
        
        return stats
    
    def reset(self) -> None:
        """Reset all metrics"""
        for key in self.metrics:
            self.metrics[key] = []

class ResourceMonitor:
    """Monitor system resources"""
    
    @staticmethod
    def get_memory_usage() -> dict:
        """Get current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024
            }
        except ImportError:
            return {"error": "psutil not installed"}
    
    @staticmethod
    def get_cpu_usage() -> float:
        """Get CPU usage percentage"""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except ImportError:
            return 0.0