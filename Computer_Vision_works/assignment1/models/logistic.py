"""Logistic regression model."""
import numpy as np
class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.
        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.
        Parameters:
            z: the input
        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        # Hint: To prevent numerical overflow, try computing the sigmoid for positive numbers and negative numbers separately.
        #       - For negative numbers, try an alternative formulation of the sigmoid function.
        w = np.zeros(z.shape)
        w[z >= 0] = 1 / (1 + np.exp(-z[z >= 0]))
        w[z < 0] = np.exp(z[z < 0]) / (1 + np.exp(z[z < 0]))
        return w

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.
        - Use the logistic regression update rule as introduced in lecture.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01. 
        - This initialization prevents the weights from starting too large,
        which can cause saturation of the sigmoid function 
        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        # pass
        self.w = np.random.uniform(-1, 1, X_train.shape[1]) * 0.01
        for epoch in range(self.epochs):
            for i in range(X_train.shape[0]):
                z = np.dot(X_train[i], self.w)
                z = self.sigmoid(z)
                error = y_train[i] - z
                self.w += self.lr * error * X_train[i]

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.
        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions
        Returns:exce
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        z = np.dot(X_test, self.w)
        z = self.sigmoid(z)
        z[z >= self.threshold] = 1
        z[z < self.threshold] = 0
        return z