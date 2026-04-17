import time
import os
import joblib
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

def save_scaler(scaler, path):
    """Save the scaler to a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)

def load_scaler(path):
    """Load the scaler from a file."""
    return joblib.load(path)

def save_model(model, path):
    """Save the model to a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)

def load_model_instance(model_class, path, **kwargs):
    """Load a model instance from a file."""
    instance = model_class(**kwargs)
    instance.load(path)
    return instance
