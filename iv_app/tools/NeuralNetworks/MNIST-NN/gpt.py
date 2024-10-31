import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

class NeuralNetwork_MNIST:
    def __init__(self, training='mnist_train.csv', testing='mnist_test.csv', use_pretrained=False, old_fit_ind=-1,
                 file_name='MNIST_W&B.json'):
        self.filename = file_name
        self.use_pretrained = use_pretrained
        self.old_fit_ind = old_fit_ind
        self.train_data = pd.read_csv(training)
        self.test_data = pd.read_csv(testing)
        self.data = np.array(self.train_data)
        self.m, self.n = self.data.shape
        np.random.shuffle(self.data)

        # Hidden data
        self.data_dev = self.data[:1000]
        self.Y_dev = self.data_dev[:, 0]
        self.X_dev = self.data_dev[:, 1:] / 255.0  # Normalize pixel values
        self.Y_dev = one_hot(self.Y_dev)

        # Real training data
        self.data_train = self.data[1000:]
        self.Y_train = self.data_train[:, 0]
        self.X_train = self.data_train[:, 1:] / 255.0  # Normalize pixel values
        self.Y_train = one_hot(self.Y_train)

        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.init_params()

        self.Z1 = None
        self.A1 = None
        self.Z2 = None
        self.A2 = None

        self.dW1 = None
        self.db1 = None
        self.dW2 = None
        self.db2 = None

        # Plotting data
        self.losses = []
        self.accuracies = []
        self.epochs = []

        # Initialize the plot
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.ax1.set_title('Loss over epochs')
        self.ax2.set_title('Accuracy over epochs')
        self.ax1.set_xlabel('Epochs')
        self.ax1.set_ylabel('Loss')
        self.ax2.set_xlabel('Epochs')
        self.ax2.set_ylabel('Accuracy')

    def init_params(self):
        if self.use_pretrained:
            return self.load_weights()
        # Xavier initialization for better convergence
        self.W1 = np.random.randn(784, 100) / np.sqrt(784)
        self.b1 = np.zeros((1, 100))
        self.W2 = np.random.randn(100, 10) / np.sqrt(100)
        self.b2 = np.zeros((1, 10))

    def fit(self, epochs=1000, lr=0.01):
        for epoch in range(epochs):
            # Forward propagation
            self.Z1 = np.dot(self.X_train, self.W1) + self.b1
            self.A1 = ReLU(self.Z1)
            self.Z2 = np.dot(self.A1, self.W2) + self.b2
            self.A2 = softmax(self.Z2)

            # Compute loss
            loss = -np.mean(self.Y_train * np.log(self.A2 + 1e-8))
            self.losses.append(loss)

            # Backward propagation
            dZ2 = self.A2 - self.Y_train
            self.dW2 = np.dot(self.A1.T, dZ2) / self.m
            self.db2 = np.sum(dZ2, axis=0, keepdims=True) / self.m
            dA1 = np.dot(dZ2, self.W2.T)
            dZ1 = dA1 * deriv_ReLU(self.Z1)
            self.dW1 = np.dot(self.X_train.T, dZ1) / self.m
            self.db1 = np.sum(dZ1, axis=0, keepdims=True) / self.m

            # Update parameters
            self.W1 -= lr * self.dW1
            self.b1 -= lr * self.db1
            self.W2 -= lr * self.dW2
            self.b2 -= lr * self.db2

            # Calculate accuracy
            if epoch % 100 == 0:
                predictions = np.argmax(self.A2, axis=1)
                labels = np.argmax(self.Y_train, axis=1)
                accuracy = np.mean(predictions == labels)
                self.accuracies.append(accuracy)
                self.epochs.append(epoch)

                # Update the plot
                self.update_plot(epoch)

    def update_plot(self, epoch):
        self.ax1.plot(self.epochs, self.losses[:len(self.epochs)], 'b')
        self.ax2.plot(self.epochs, self.accuracies, 'r')
        plt.pause(0.01)
        self.fig.canvas.draw()
        print(f"Epoch {epoch}, Loss: {self.losses[-1]}, Accuracy: {self.accuracies[-1]}")

    def evaluate(self):
        self.Z1 = np.dot(self.X_dev, self.W1) + self.b1
        self.A1 = ReLU(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = softmax(self.Z2)
        predictions = np.argmax(self.A2, axis=1)
        labels = np.argmax(self.Y_dev, axis=1)
        accuracy = np.mean(predictions == labels)
        return accuracy

    def load_weights(self):
        with open(self.filename, 'r') as f:
            weights = json.load(f)
        self.W1 = np.array(weights['W1'])
        self.b1 = np.array(weights['b1'])
        self.W2 = np.array(weights['W2'])
        self.b2 = np.array(weights['b2'])

    def save_weights(self):
        weights = {
            'W1': self.W1.tolist(),
            'b1': self.b1.tolist(),
            'W2': self.W2.tolist(),
            'b2': self.b2.tolist()
        }
        with open(self.filename, 'w') as f:
            json.dump(weights, f)

# Ensure to call plt.show() at the end of your script to display the plot
if __name__ == "__main__":
    nn = NeuralNetwork_MNIST()
    nn.fit(epochs=1000, lr=0.01)
    accuracy = nn.evaluate()
    print(f"Validation Accuracy: {accuracy}")
    plt.show()