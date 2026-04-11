import time
from functools import wraps

# Global dictionary to store aggregate times across all files
PROFILE_STATS = {}

def timeit(name=None):
    """Decorator to time low-level function calls and aggregate the results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            
            func_name = name or func.__name__
            if func_name not in PROFILE_STATS:
                PROFILE_STATS[func_name] = {'count': 0, 'total_time': 0.0}
            
            PROFILE_STATS[func_name]['count'] += 1
            PROFILE_STATS[func_name]['total_time'] += elapsed
            return result
        return wrapper
    return decorator

class TimerBlock:
    """Context manager to time high-level blocks of code (e.g., loops or pipeline steps)."""
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        if self.name not in PROFILE_STATS:
            PROFILE_STATS[self.name] = {'count': 0, 'total_time': 0.0}
        
        PROFILE_STATS[self.name]['count'] += 1
        PROFILE_STATS[self.name]['total_time'] += elapsed

def print_profile_stats():
    """Prints a clean, formatted table of all profiled sections."""
    print("\n" + "="*60)
    print(f"{'PIPELINE PROFILING RESULTS':^60}")
    print("="*60)
    print(f"{'Function / Logic Block':<35} | {'Calls':<8} | {'Total Time (s)':<10}")
    print("-" * 60)
    # Sort by total time descending
    for name, stats in sorted(PROFILE_STATS.items(), key=lambda x: x[1]['total_time'], reverse=True):
        print(f"{name:<35} | {stats['count']:<8} | {stats['total_time']:.4f}")
    print("="*60 + "\n")