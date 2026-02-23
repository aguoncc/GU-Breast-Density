from functools import wraps
import errno
import os
import signal
import platform
import sys # adding because signal.SIGALRM doesnt exist for windows

class TimeoutError(Exception):
    pass

# # editing the timeout function to wrapper(func) instead
def timeout(seconds=60, error_message = "Function timed out"):
    """
    Docstring for timeout
    
    :param seconds: Description
    :param error_message: Description
    """
    system = platform.system()

    def decorator(func):
        if system == "Windows":
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        
        else:
            def _handle_timeout(signum, frame):
                raise TimeoutError(error_message)
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                signal.signal(signal.SIGALRM, _handle_timeout)
                signal.alarm(seconds)
                try:
                    return func(*args, **kwargs)
                finally:
                    signal.alarm(0)
            
            return wrapper
    return decorator