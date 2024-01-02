import time
import psutil
import os

class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0

    def update_frame_count(self):
        self.frame_count += 1

    def get_fps(self):
        current_time = time.time()
        fps = self.frame_count / (current_time - self.start_time)
        return fps

def measure_execution_time(func):
    """
    Decorator function to measure the execution time of a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def get_memory_usage():
    """
    Returns the current memory usage of the application.
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss  # rss = Resident Set Size

def get_cpu_usage():
    """
    Returns the current CPU usage as a percentage.
    """
    return psutil.cpu_percent(interval=1)

# Example usage of decorator
@measure_execution_time
def some_function_to_measure():
    # Function logic here
    pass

# Example usage of PerformanceMonitor
performance_monitor = PerformanceMonitor()
# ... within a loop or repeated operation ...
performance_monitor.update_frame_count()
print("Current FPS:", performance_monitor.get_fps())
