import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split

def ReLU(Z):
    """
    Applies the Rectified Linear Unit (ReLU) activation function.
    ReLU(x) = max(0, x)

    :param Z: Input array
    :return: Array with ReLU applied element-wise
    """
    return np.maximum(0, Z)


def deriv_ReLU(Z):
    """
    Computes the derivative of the ReLU function.
    The derivative is 1 for x > 0, and 0 for x <= 0.

    :param Z: Input array
    :return: Array with derivative of ReLU applied element-wise
    """
    return Z > 0  # Returns a boolean array, which is implicitly converted to 0s and 1s


def softmax(Z):
    """
    Applies the softmax function to convert raw scores to probabilities.
    softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j

    :param Z: 2D array of shape (batch_size, num_classes)
    :return: 2D array of same shape with softmax applied row-wise
    """
    # Subtract max for numerical stability (prevents overflow)
    Z = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=1, keepdims=True)


def one_hot(column):
    """
    Converts a 1D array of class labels to a 2D one-hot encoded array.

    :param column: 1D array of class labels (0-9 for MNIST)
    :return: 2D array where each row is a one-hot vector
    """
    new_yt = np.zeros((len(column), 10))
    for i, num in enumerate(column):
        new_yt[i, num] = 1
    return new_yt


class NeuralNetwork_MNIST:
    def __init__(self, training='mnist_train.csv', testing='mnist_test.csv', use_pretrained=False, old_fit_ind=-1,
                 json_file='three_layer_MNIST_W&B.json'):
        self.json_file = json_file
        self.use_pretrained = use_pretrained
        self.old_fit_ind = old_fit_ind

        if not self.use_pretrained:
            # Load and preprocess data
            self.train_data = pd.read_csv(training)
            self.test_data = pd.read_csv(testing)
            self.data = np.array(self.train_data)
            self.m, self.n = self.data.shape  # m: number of examples, n: number of features + 1 (label)
            np.random.shuffle(self.data)

            # Split data into dev and train sets
            self.data_dev = self.data[:5000]
            self.Y_dev = one_hot(self.data_dev[:, 0])
            self.X_dev = self.data_dev[:, 1:] / 255.0  # Normalize pixel values to [0, 1]

            self.data_train = self.data[5000:]
            self.Y_train = one_hot(self.data_train[:, 0])
            self.X_train = self.data_train[:, 1:] / 255.0  # Normalize pixel values to [0, 1]

        # Initialize weights and biases
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.W3 = None
        self.b3 = None
        self.init_params()

        # Placeholders for intermediate values during forward pass
        self.Z1 = None
        self.A1 = None
        self.Z2 = None
        self.A2 = None
        self.Z3 = None
        self.A3 = None

        # Placeholders for gradients
        self.dW1 = None
        self.db1 = None
        self.dW2 = None
        self.db2 = None
        self.dW3 = None
        self.db3 = None

    def init_params(self):
        """
        Initialize random weights and biases.
        W1: (784, 100) - connects input layer (784 nodes) to hidden layer (100 nodes)
        b1: (1, 100) - bias for hidden layer
        W2: (100, 100) -
        b2: (1, 100)
        W3: (100, 10) - connects hidden layer (100 nodes) to output layer (10 nodes)
        b3: (1, 10) - bias for output layer
        """
        if self.use_pretrained:
            return self.load_weights()

        # He initialization for weights
        self.W1 = np.random.randn(784, 100) / np.sqrt(784)
        self.b1 = np.zeros((1, 100))
        self.W2 = np.random.randn(100, 100) / np.sqrt(100)
        self.b2 = np.zeros((1, 100))

        self.W3 = np.random.randn(100, 10) / np.sqrt(100)
        self.b3 = np.zeros((1, 10))

    def save_weights(self, overwrite=True):
        """
        Save weights and biases to a JSON file.
        """
        weights_dict = {
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
            "W3": self.W3.tolist(),
            "b3": self.b3.tolist()
        }

        if overwrite or not os.path.exists(self.json_file):
            data = [weights_dict]
        else:
            with open(self.json_file, 'r') as f:
                data = json.load(f)
            data.append(weights_dict)

        with open(self.json_file, 'w') as f:
            json.dump(data, f)

    def load_weights(self):
        """
        Load weights and biases from a JSON file.
        """
        if os.path.exists(self.json_file):
            with open(self.json_file, 'r') as f:
                data = json.load(f)

            weights_dict = data[-1]  # Get the last set of weights

            self.W1 = np.array(weights_dict["W1"])
            self.b1 = np.array(weights_dict["b1"])
            self.W2 = np.array(weights_dict["W2"])
            self.b2 = np.array(weights_dict["b2"])
            self.W3 = np.array(weights_dict["W3"])
            self.b3 = np.array(weights_dict["b3"])
        else:
            raise FileNotFoundError(f"File '{self.json_file}' not found.")

    def feed_forward_train(self):
        """
        Perform forward propagation on the training data.
        """
        # First layer
        self.Z1 = np.dot(self.X_train, self.W1) + self.b1  # Shape: (m, 100)
        self.A1 = ReLU(self.Z1)  # Shape: (m, 100)

        # Second layer
        self.Z2 = np.dot(self.A1, self.W2) + self.b2  # Shape: (m, 100)
        self.A2 = ReLU(self.Z2)  # Shape: (m, 100)

        # third (output) layer
        self.Z3 = np.dot(self.A2, self.W3) + self.b3  # Shape: (m, 10)
        self.A3 = softmax(self.Z3)  # Shape: (m, 10)

    def back_prop(self):
        """
        Perform backpropagation to compute gradients.
        """
        m = self.X_train.shape[0]

        # Output layer
        dZ3 = self.A3 - self.Y_train  # Shape: (m, 10)
        self.dW3 = (1 / m) * np.dot(self.A2.T, dZ3)  # Shape: (100, 10)
        self.db3 = (1 / m) * np.sum(dZ3, axis=0, keepdims=True)  # Shape: (1, 10)

        # Hidden layer
        dZ2 = np.dot(dZ3, self.W3.T) * deriv_ReLU(self.Z2)  # Shape: (m, 100)
        self.dW2 = (1 / m) * np.dot(self.A1.T, dZ2)  # Shape: (784, 100)
        self.db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)  # Shape: (1, 100)

        # Hidden layer
        dZ1 = np.dot(dZ2, self.W2.T) * deriv_ReLU(self.Z1)  # Shape: (m, 100)
        self.dW1 = (1 / m) * np.dot(self.X_train.T, dZ1)  # Shape: (784, 100)
        self.db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)  # Shape: (1, 100)

    def gradient_descent(self, iterations, alpha, save_over=.9):
        """
        Perform gradient descent to optimize the parameters.

        :param iterations: Number of training iterations
        :param alpha: Learning rate
        :param save_over: Accuracy threshold to save weights
        """
        for i in range(iterations):
            self.feed_forward_train()
            self.back_prop()

            # Update parameters
            self.W1 -= alpha * self.dW1
            self.b1 -= alpha * self.db1
            self.W2 -= alpha * self.dW2
            self.b2 -= alpha * self.db2
            self.W3 -= alpha * self.dW3
            self.b3 -= alpha * self.db3

            # Print progress every 100 iterations
            if i % 100 == 0:
                loss = self.calculate_loss()
                accuracy = self.get_accuracy(self.X_dev, self.Y_dev)

                if accuracy >= save_over:
                    self.save_weights()

                print(f"Iteration {i}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")

    def calculate_loss(self):
        """
        Calculate the cross-entropy loss.
        """
        m = self.Y_train.shape[0]
        log_probs = np.log(self.A3 + 1e-8)  # Add small epsilon to avoid log(0)
        loss = -np.sum(self.Y_train * log_probs) / m
        return loss

    def get_accuracy(self, X, Y):
        """
        Calculate the accuracy of predictions.

        :param X: Input data
        :param Y: True labels (one-hot encoded)
        :return: Accuracy as a fraction
        """
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = ReLU(Z1)

        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = ReLU(Z2)

        Z3 = np.dot(A2, self.W3) + self.b3
        A3 = softmax(Z3)
        predictions = np.argmax(A3, axis=1)
        return np.mean(predictions == np.argmax(Y, axis=1))

    def predict(self, X):
        """
        Make predictions on new data.

        :param X: Input data
        :return: Predicted class labels
        """
        X = X / 255.0  # Normalize pixel values
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = ReLU(Z1)

        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = ReLU(Z2)

        Z3 = np.dot(A2, self.W3) + self.b3
        A3 = softmax(Z3)
        return [np.argmax(A3, axis=1), 100*round(max(max(A3)), 4)]


if __name__ == "__main__":
    # nn = NeuralNetwork_MNIST(use_pretrained=True)
    # # nn.gradient_descent(iterations=100, alpha=0.09)
    #
    # # Make predictions on test data
    # test_data = pd.read_csv('mnist_test.csv').values
    # x_ind = int(np.random.random() * 950)
    #
    # X_test = test_data[x_ind:x_ind+27, 1:]
    # answers = test_data[x_ind:x_ind+27, 0]
    #
    # pred = nn.predict(X_test)
    # for i in range(len(pred)):
    #     print(f"Value: {answers[i]}, Prediction: {pred[i]}")

    nn = NeuralNetwork_MNIST()
    # nn.gradient_descent(2973, alpha=.01)
    # nn.save_weights()
















    # NEED to show distribution to user, not just max pick















