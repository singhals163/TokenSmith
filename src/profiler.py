import time
from functools import wraps

PROFILE_STATS = {}


def timeit(name=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start

            func_name = name or func.__name__
            stats = PROFILE_STATS.setdefault(func_name, {"count": 0, "total_time": 0.0})
            stats["count"] += 1
            stats["total_time"] += elapsed
            return result
        return wrapper
    return decorator


class TimerBlock:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        stats = PROFILE_STATS.setdefault(self.name, {"count": 0, "total_time": 0.0})
        stats["count"] += 1
        stats["total_time"] += elapsed


def print_profile_stats(filepath=None):
    output = []
    output.append("\n" + "=" * 60)
    output.append(f"{'PIPELINE PROFILING RESULTS':^60}")
    output.append("=" * 60)
    output.append(f"{'Function / Logic Block':<35} | {'Calls':<8} | {'Total Time (s)':<10}")
    output.append("-" * 60)

    for name, stats in sorted(PROFILE_STATS.items(), key=lambda x: x[1]["total_time"], reverse=True):
        output.append(f"{name:<35} | {stats['count']:<8} | {stats['total_time']:.4f}")
    output.append("=" * 60 + "\n")

    final_text = "\n".join(output)
    if filepath:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(final_text)
    else:
        print(final_text)
