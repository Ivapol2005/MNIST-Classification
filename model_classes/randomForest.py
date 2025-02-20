from .MnistClassifierInterface import MnistClassifierInterface

from sklearn.ensemble import RandomForestClassifier
import numpy as np

class RandomForestMnistClassifier(MnistClassifierInterface):
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    def train(self, X, y):
        X = X.reshape(X.shape[0], -1)
        self.model.fit(X, y)

    def predict(self, X_test):
        X_test = X_test.reshape(X_test.shape[0], -1)
        return self.model.predict(X_test)
