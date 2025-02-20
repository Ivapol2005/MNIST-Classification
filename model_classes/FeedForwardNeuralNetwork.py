from .MnistClassifierInterface import MnistClassifierInterface

import tensorflow as tf
# from tensorflow.keras.datasets import mnist

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

import numpy as np

class FeedForwardNNClassifier(MnistClassifierInterface):
    def __init__(self):
        self.model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])
        
    def train(self, X, y):
        X = X.reshape(-1, 28, 28, 1).astype('float32') # / 255.0
        y = to_categorical(y, 10)
        
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(X, y, epochs=10, batch_size=32, validation_split=0.1)
        
    def predict(self, X_test):
        X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') # / 255.0
        predictions = self.model.predict(X_test)
        return np.argmax(predictions, axis=-1)