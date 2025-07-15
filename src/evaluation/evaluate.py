import numpy as np

def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def SMAPE(y_true, y_pred):
    diff = np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))
    return 2.0 * np.mean(diff, axis=-1) * 100

def R2(y_true, y_pred):
    SS_res = np.sum(np.square(np.subtract(y_true, y_pred)))
    SS_tot = np.sum(np.square(np.subtract(y_true, np.mean(y_true))))
    return  1 - SS_res / SS_tot

def MASE(y_true, y_pred):
    naive_error = np.mean(np.abs(y_true[1:] - y_true[:-1]))
    return np.mean(np.abs(y_true - y_pred)) / naive_error

def RMSSE(y_true, y_pred):
    naive_error = np.sqrt(np.mean(np.square(y_true[1:] - y_true[:-1])))
    return np.sqrt(np.mean(np.square(y_true - y_pred))) / naive_error

def MDA(y_true, y_pred):
    return np.mean(np.equal(np.sign(y_true[1:] - y_true[:-1]), np.sign(y_pred[1:] - y_pred[:-1]))) * 100

def evaluate(model, metrics, X_test, y_test):
    results = {}
    for metric_name, metric in metrics.items():
        metric_value = metric(y_test, model.predict(X_test))
        results[metric_name] = metric_value
    return results
