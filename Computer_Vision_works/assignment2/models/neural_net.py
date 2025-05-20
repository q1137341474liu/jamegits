"""Neural network model."""
from typing import Sequence
import numpy as np
class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C. 
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last. 
    The outputs of the last fully-connected layer are passed through
    a sigmoid. 
    """
    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        opt: str,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
            opt: option for using "SGD" or "Adam" optimizer (Adam is Extra Credit)
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers
        self.opt = opt
        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]
        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(sizes[i - 1], sizes[i]) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])
            # TODO: (Extra Credit) You may set parameters for Adam optimizer here
    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        # TODO: implement me
        return X @ W + b
    def linear_grad(self, W: np.ndarray, X: np.ndarray, de_dz: np.ndarray) -> np.ndarray:
        """Gradient of linear layer
        Parameters:
            W: the weight matrix
            X: the input data
            de_dz: the gradient of loss
        Returns:
            de_dw, de_db, de_dx
            where
                de_dw: gradient of loss with respect to W
                de_db: gradient of loss with respect to b
                de_dx: gradient of loss with respect to X
        """
        # TODO: implement me
        de_dw = X.T @ de_dz
        de_db = np.sum(de_dz, axis = 0)
        de_dx = de_dz @ W
        return de_dw, de_db, de_dx
    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        # TODO: implement me
        return np.maximum(0, X)
    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
         # TODO: implement me
        return np.where(X > 0, 1, 0)
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        # TODO ensure that this is numerically stable
        w = np.zeros(x.shape)
        w[x >= 0] = 1 / (1 + np.exp(-x[x >= 0]))
        w[x < 0] = np.exp(x[x < 0]) / (1 + np.exp(x[x < 0]))
        return w
    def sigmoid_grad(self, X: np.ndarray) -> np.ndarray:
        # TODO implement this
        return self.sigmoid(X) * (1 - self.sigmoid(X))
    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        return np.mean((y - p) ** 2)
    def mse_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        return 2 * (p - y) / len(y)
    def mse_sigmoid_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this                    
        return self.sigmoid_grad(p) * self.mse_grad(y, p)
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) 
        """
        self.outputs = {}
        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.mse in here.
        # tmp = X.copy()
        # self.outputs["o0"] = tmp
        # for i in range(1, self.num_layers + 1):
        #     tmp = self.linear(self.params["W" + str(i)], tmp, self.params["b" + str(i)])
        #     self.outputs["layer" + str(i)] = tmp
        #     if i != self.num_layers:
        #         tmp = self.relu(tmp)
        #     else:
        #         tmp = self.sigmoid(tmp)
        #     self.outputs["o" + str(i)] = tmp
        # return tmp
        predictions = []
        for start in range(0, X.shape[0], 32):
            end = min(start + 32, X.shape[0])
            tmp = X[start:end].copy()
            batch_outputs = {"o0": tmp}
            for i in range(1, self.num_layers + 1):
                tmp = self.linear(self.params["W" + str(i)], tmp, self.params["b" + str(i)])
                batch_outputs["layer" + str(i)] = tmp
                if i != self.num_layers:
                    tmp = self.relu(tmp)
                else:
                    tmp = self.sigmoid(tmp)
                batch_outputs["o" + str(i)] = tmp
            for key, value in batch_outputs.items():
                if key not in self.outputs:
                    self.outputs[key] = []
                self.outputs[key].append(value)
            predictions.append(tmp)
        for key in self.outputs:
            self.outputs[key] = np.vstack(self.outputs[key])
        return np.vstack(predictions)
    def backward(self, y: np.ndarray) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets
        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        # TODO: implement me. You'll want to store the gradient of each
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.linear_grad,
        # self.relu_grad, and self.softmax_grad if it helps organize your code.
        # loss = self.mse(y, self.outputs["o" + str(self.num_layers)])
        # ez = -2 / 3 * (y - self.outputs["o" + str(self.num_layers)])
        # for i in range(self.num_layers, 0, -1):
        #     if i == self.num_layers:
        #         cz = self.sigmoid_grad(self.outputs["layer" + str(i)]) * ez
        #     else:
        #         cz = self.relu_grad(self.outputs["layer" + str(i)]) * ez
        #     self.gradients["W" + str(i)] = self.outputs["o" + str(i - 1)].T @ cz / len(y)
        #     self.gradients["b" + str(i)] = np.mean(cz, axis = 0)
        #     ez = cz @ self.params["W" + str(i)].T
        # return loss
        total_loss = 0.0
        for start in range(0, y.shape[0], 32):
            end = min(start + 32, y.shape[0])
            y_batch = y[start:end]
            output_batch = self.outputs["o" + str(self.num_layers)][start:end]
            loss = self.mse(y_batch, output_batch)
            total_loss += loss * (end - start)
            ez = -2 / (end - start) * (y_batch - output_batch)
            for i in range(self.num_layers, 0, -1):
                if i == self.num_layers:
                    cz = self.sigmoid_grad(self.outputs["layer" + str(i)][start:end]) * ez
                else:
                    cz = self.relu_grad(self.outputs["layer" + str(i)][start:end]) * ez
                if "W" + str(i) not in self.gradients:
                    self.gradients["W" + str(i)] = np.zeros_like(self.params["W" + str(i)])
                    self.gradients["b" + str(i)] = np.zeros_like(self.params["b" + str(i)])
                self.gradients["W" + str(i)] += self.outputs["o" + str(i - 1)][start:end].T @ cz / (end - start)
                self.gradients["b" + str(i)] += np.mean(cz, axis=0)
                ez = cz @ self.params["W" + str(i)].T
        return total_loss / y.shape[0]
    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
        """
        if self.opt == 'SGD':
            # TODO: implement SGD optimizer here
            for i in range(1, self.num_layers + 1):
                self.params["W" + str(i)] -= lr * self.gradients["W" + str(i)]
                self.params["b" + str(i)] -= lr * self.gradients["b" + str(i)]
            pass
        elif self.opt == 'Adam':
            # TODO: (Extra credit) implement Adam optimizer here
            pass
        else:
            raise NotImplementedError