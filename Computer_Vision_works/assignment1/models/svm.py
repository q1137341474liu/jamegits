"""Support Vector Machine (SVM) model."""
import numpy as np
class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.
        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.b = None
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class
    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.
        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.
        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C
        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        # TODO: implement me
        v = X_train @ self.w + self.b
        v_correct = v[np.arange(X_train.shape[0]), y_train][:, np.newaxis]
        mar = np.maximum(0, v - v_correct + 1)
        # mar = np.where(mar > 0, 1, 0)
        mar[np.arange(X_train.shape[0]), y_train] = 0
        m = (mar > 0).astype(float)
        m[np.arange(X_train.shape[0]), y_train] = -m.sum(axis = 1)
        return X_train.T @ m / X_train.shape[0] + self.w * self.reg_const, np.mean(m, axis = 0)
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.
        Hint: operate on mini-batches of data for SGD.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01. This scaling prevents overly large initial weights,
        which can adversely affect training.
        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        self.w = 0.01 * np.random.uniform(-1, 1, (X_train.shape[1], self.n_class))
        self.b = np.zeros(self.n_class)
        vw = np.zeros_like(self.w)
        vb = np.zeros_like(self.b)
        self.mean = np.mean(X_train, axis = 0)
        self.std = np.std(X_train, axis = 0)
        self.std[self.std == 0] = 1
        X_train = (X_train - self.mean) / self.std
        for _ in range(self.epochs):
            for i in range(0, X_train.shape[0], 128):
                gw, gb = self.calc_gradient(X_train[i:i + 128], y_train[i:i + 128])
                vw = 0.9 * vw - self.lr * gw
                vb = 0.9 * vb - self.lr * gb
                self.w += vw
                self.b += vb
            self.lr *= 0.95
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
        v = (X_test - self.mean) / self.std @ self.w + self.b
        return np.argmax(v, axis = 1)