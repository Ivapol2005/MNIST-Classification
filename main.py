from abc import ABC, abstractmethod


"""
class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self, X, y):
        # method for training
        pass

    @abstractmethod
    def predict(self, X):
        # method for prediction
        pass
"""


from model_classes.randomForest import RandomForestMnistClassifier
from model_classes.FeedForwardNeuralNetwork import FeedForwardNNClassifier
from model_classes.ConvolutionalNeuralNetwork import ConvolutionalNNClassifier

    
class MnistClassifier:
    def __init__(self, algorithm="rf"):
        # makes model depending on algorithm
        if algorithm == "rf":
            self.classifier = RandomForestMnistClassifier()
        elif algorithm == "nn":
            self.classifier = FeedForwardNNClassifier()
        elif algorithm == "cnn":
            self.classifier = ConvolutionalNNClassifier()
        else:
            raise ValueError(
                "Possible algorithms:\n"
                "'rf' - Random Forest\n"
                "'nn' - Feed Forward Neural Network\n"
                "'cnn' - Convolutional Neural Network"
            )
    def train(self, X, y):
        self.classifier.train(X, y)

    def predict(self, X_test):
        predictions = self.classifier.predict(X_test)
        return predictions


    
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

mnist_rf = MnistClassifier(algorithm="rf")
mnist_rf.train(X_train, y_train)
y_pred_rf = mnist_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

mnist_nn = MnistClassifier(algorithm="nn")
mnist_nn.train(X_train, y_train)
y_pred_nn = mnist_nn.predict(X_test)
accuracy_nn = accuracy_score(y_test, y_pred_nn)

mnist_cnn = MnistClassifier(algorithm="cnn")
mnist_cnn.train(X_train, y_train)
y_pred_cnn = mnist_cnn.predict(X_test)
accuracy_cnn = accuracy_score(y_test, y_pred_cnn)

print()
print(f"Random Forest Accuracy on MNIST: {accuracy_rf:.4f}")
print(f"Feed Forward Neural Network Accuracy on MNIST: {accuracy_nn:.4f}")
print(f"Convolutional Neural Network on MNIST: {accuracy_cnn:.4f}")
