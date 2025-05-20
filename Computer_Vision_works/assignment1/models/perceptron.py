import numpy as np
class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int, decay_factor: float = 0.1):
        """Initialize a new classifier.
        Parameters:
            n_class: the number of classes
            lr: the initial learning rate
            epochs: the number of epochs to train for
            decay_factor: factor by which to decay the learning rate each epoch
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.decay_factor = decay_factor

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.
        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        self.w = 0.01 * np.random.uniform(-1, 1, (X_train.shape[1], self.n_class))
        for epoch in range(self.epochs):
            lr = self.lr * (self.decay_factor ** epoch)
            for i in range(X_train.shape[0]):
                p = np.dot(X_train[i], self.w)
                label = np.argmax(p)
                if label != y_train[i]:
                    self.w[:, y_train[i]] += lr * X_train[i]
                    self.w[:, label] -= lr * X_train[i]

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.
        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions
        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        scores = np.dot(X_test, self.w)
        return np.argmax(scores, axis=1)