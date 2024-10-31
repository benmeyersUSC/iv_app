import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from NEURAL_NETWORKS.graph_28 import plot_image_from_vector

def ReLU(Z):
    return np.maximum(0, Z)

def deriv_ReLU(Z):
    return Z > 0

def softmax(Z):
    Z = Z - np.max(Z, axis=1, keepdims=True)  # Subtract max for numerical stability
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def one_hot(column):
    new_yt = np.zeros((len(column), 10))
    for i, num in enumerate(column):
        new_yt[i, num] = 1
    return new_yt

class NeuralNetwork_F_MNIST_expanded:
    def __init__(self, training='fashion-mnist_train.csv', testing='fashion-mnist_test.csv', use_pretrained=False,
                 old_fit_ind=-1,
                 file_name='F_MNIST_W&B.json'):
        self.filename = file_name
        self.use_pretrained = use_pretrained
        self.old_fit_ind = old_fit_ind
        self.train_data = pd.read_csv(training)
        self.test_data = pd.read_csv(testing)
        self.data = np.array(self.train_data)
        self.m, self.n = self.data.shape
        np.random.shuffle(self.data)

        # Hidden data
        self.data_dev = self.data[:10000]
        self.Y_dev = self.data_dev[:, 0]
        self.X_dev = self.data_dev[:, 1:] / 255.0  # Normalize pixel values
        self.Y_dev = one_hot(self.Y_dev)

        # Real training data
        self.data_train = self.data[10000:]
        self.Y_train = self.data_train[:, 0]
        self.X_train = self.data_train[:, 1:] / 255.0  # Normalize pixel values
        self.Y_train = one_hot(self.Y_train)

        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.W3 = None
        self.b3 = None
        self.init_params()

        self.Z1 = None
        self.A1 = None
        self.Z2 = None
        self.A2 = None
        self.Z3 = None
        self.A3 = None

        self.dW1 = None
        self.db1 = None
        self.dW2 = None
        self.db2 = None
        self.dW3 = None
        self.db3 = None

    def init_params(self):
        if self.use_pretrained:
            return self.load_weights()
        # Xavier initialization for better convergence
        self.W1 = np.random.randn(784, 400) / np.sqrt(784)
        self.b1 = np.zeros((1, 400))
        self.W2 = np.random.randn(400, 100) / np.sqrt(400)
        self.b2 = np.zeros((1, 100))
        self.W3 = np.random.randn(100, 10) / np.sqrt(100)
        self.b3 = np.zeros((1, 10))

    def feed_forward_train(self):
        self.Z1 = self.X_train.dot(self.W1) + self.b1
        self.A1 = ReLU(self.Z1)
        self.Z2 = self.A1.dot(self.W2) + self.b2
        self.A2 = ReLU(self.Z2)
        self.Z3 = self.A2.dot(self.W3) + self.b3
        self.A3 = softmax(self.Z3)

    def back_prop(self):
        m = self.X_train.shape[0]

        # Output layer
        dZ3 = self.A3 - self.Y_train
        self.dW3 = (1 / m) * np.dot(self.A2.T, dZ3)
        self.db3 = (1 / m) * np.sum(dZ3, axis=0, keepdims=True)

        # Hidden layer
        dZ2 = np.dot(dZ3, self.W3.T) * deriv_ReLU(self.Z2)
        self.dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
        self.db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

        dZ1 = np.dot(dZ2, self.W2.T) * deriv_ReLU(self.Z1)
        self.dW1 = (1 / m) * np.dot(self.X_train.T, dZ1)
        self.db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

    def gradient_descent(self, iterations, alpha, save_over=.75):
        for i in range(iterations):
            # Forward propagation
            self.feed_forward_train()

            # Backpropagation
            self.back_prop()

            # Update parameters
            self.W1 -= alpha * self.dW1
            self.b1 -= alpha * self.db1
            self.W2 -= alpha * self.dW2
            self.b2 -= alpha * self.db2
            self.W3 -= alpha * self.dW3
            self.b3 -= alpha * self.db3



            # Optional: Print loss every 100 iterations
            if i % 100 == 0:
                loss = self.calculate_loss()

                accuracy = self.get_accuracy(self.X_dev, self.Y_dev)

                if accuracy >= save_over:
                    self.save_weights()

                print(f"Iteration {i}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")

    def calculate_loss(self):
        m = self.Y_train.shape[0]
        log_probs = np.log(self.A3 + 1e-8)  # Add small epsilon to avoid log(0)
        loss = -np.sum(self.Y_train * log_probs) / m
        return loss

    def get_accuracy(self, X, Y):
        Z1 = X.dot(self.W1) + self.b1
        A1 = ReLU(Z1)

        Z2 = A1.dot(self.W2) + self.b2
        A2 = ReLU(Z2)


        Z3 = A2.dot(self.W3) + self.b3
        A3 = softmax(Z3)
        predictions = np.argmax(A3, axis=1)
        return np.mean(predictions == np.argmax(Y, axis=1))

    def predict(self, X):
        X = X / 255.0  # Normalize pixel values
        Z1 = X.dot(self.W1) + self.b1
        A1 = ReLU(Z1)
        Z2 = A1.dot(self.W2) + self.b2
        A2 = ReLU(Z2)


        Z3 = A2.dot(self.W3) + self.b3
        A3 = softmax(Z3)
        return np.argmax(A3, axis=1)

    def save_weights(self, overwrite=True):
        weights_dict = {
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
            "W3": self.W3.tolist(),
            "b3": self.b3.tolist()
        }

        if overwrite or not os.path.exists(self.filename):
            data = [weights_dict]
        else:
            with open(self.filename, 'r') as f:
                data = json.load(f)
            data.append(weights_dict)

        with open(self.filename, 'w') as f:
            json.dump(data, f)

    def load_weights(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                data = json.load(f)

            # Get the last set of weights from the loaded data
            weights_dict = data[-1]

            # Convert lists back to numpy arrays
            self.W1 = np.array(weights_dict["W1"])
            self.b1 = np.array(weights_dict["b1"])
            self.W2 = np.array(weights_dict["W2"])
            self.b2 = np.array(weights_dict["b2"])
            self.W3 = np.array(weights_dict["W3"])
            self.b3 = np.array(weights_dict["b3"])
        else:
            raise FileNotFoundError(f"File '{self.filename}' not found.")





# Usage
nn = NeuralNetwork_F_MNIST_expanded()
nn.gradient_descent(iterations=200, alpha=0.18)

# After training, you can make predictions
test_data = pd.read_csv('fashion-mnist_test.csv').values
X_test = test_data[900:927, 1:]
answers = test_data[900:927, 0]

plot_image_from_vector(X_test[9])

print(answers)
predictions = nn.predict(X_test)

print(predictions)










