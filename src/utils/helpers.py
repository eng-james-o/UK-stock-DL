import time
from functools import wraps

def watch(function):
    '''Decorator to time the execution of a function'''
    @wraps(function)
    def watch_(*args, **kwargs):
        start = time.time()
        ret = function(*args, **kwargs)
        end = time.time()
        elapsed = end - start
        print(f'time: {elapsed}')
        return ret, elapsed
    return watch_
