from .MnistClassifierInterface import MnistClassifierInterface

import tensorflow as tf
# from tensorflow.keras.datasets import mnist

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

import numpy as np

class ConvolutionalNNClassifier(MnistClassifierInterface):
    def __init__(self):
        self.model = tf.keras.models.Sequential([
            Conv2D(32, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),

            Conv2D(64, kernel_size=(3,3), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),

            Flatten(),
            Dense(128),
            Dense(10, activation='softmax')
        ])
        
    def train(self, X, y):
        X = X.reshape(-1, 28, 28, 1).astype('float32') # / 255.0
        y = to_categorical(y, 10)
        
        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(X, y, epochs=10)
        
    def predict(self, X_test):
        X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') # / 255.0
        predictions = self.model.predict(X_test)
        return np.argmax(predictions, axis=-1)