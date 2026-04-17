class BaseModel:
    def __init__(self):
        self.model = None

    def fit(self, X, y, **kwargs):
        raise NotImplementedError

    def predict(self, X, **kwargs):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError
