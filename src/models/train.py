from sklearn.model_selection import train_test_split

def split_data(X_data, y_data, train=0.7, test=0.2):
    val = (1 - (train + test))/train
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test
