import json
import os
import random
import matplotlib.pyplot as plt
import numpy as np
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

class NeuralNetwork_MNIST:
    def __init__(self, training='mnist_train.csv', testing='mnist_test.csv', use_pretrained=False, old_fit_ind=-1,
                 json_file='MNIST_W&B.json'):
        self.json_file = json_file
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
        # time/iter series of biases
        self.b11_over_time = []
        self.b21_over_time = []
        self.init_params()

        self.Z1 = None
        self.A1 = None
        self.Z2 = None
        self.A2 = None

        self.dW1 = None
        self.db1 = None
        self.dW2 = None
        self.db2 = None


    def init_params(self):
        if self.use_pretrained:
            return self.load_weights()
        # Xavier initialization for better convergence
        self.W1 = np.random.randn(784, 100) / np.sqrt(784)
        self.b1 = np.zeros((1, 100))
        self.W2 = np.random.randn(100, 10) / np.sqrt(100)
        self.b2 = np.zeros((1, 10))

        self.b11_over_time.append(self.b1[0][0]*1000)
        self.b21_over_time.append(self.b2[0][0]*1000)

    def feed_forward_train(self):
        self.Z1 = self.X_train.dot(self.W1) + self.b1
        self.A1 = ReLU(self.Z1)
        self.Z2 = self.A1.dot(self.W2) + self.b2
        self.A2 = softmax(self.Z2)

    def back_prop(self):
        m = self.X_train.shape[0]

        # Output layer
        dZ2 = self.A2 - self.Y_train
        self.dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
        self.db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

        # Hidden layer
        dZ1 = np.dot(dZ2, self.W2.T) * deriv_ReLU(self.Z1)
        self.dW1 = (1 / m) * np.dot(self.X_train.T, dZ1)
        self.db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

    def gradient_descent(self, iterations, alpha, save_over=.9):
        for i in range(iterations):
            # Forward propagation
            self.feed_forward_train()

            # Backpropagation
            self.back_prop()

            # Update parameters
            self.W1 -= alpha * self.dW1
            # print(f"B11 pre {self.b1}")
            self.b1 -= alpha * self.db1
            # print(f"b1 first: {self.b1[0][0]}")
            # print(f"B11 post {self.b1}")
            self.W2 -= alpha * self.dW2
            self.b2 -= alpha * self.db2

            if i % 10 == 0:
                self.b11_over_time.append(self.b1[0][int(random.random()*10)]*1000)
                self.b21_over_time.append(self.b2[0][int(random.random()*10)]*1000)



            # Optional: Print loss every 100 iterations
            if i % 100 == 0:
                loss = self.calculate_loss()

                accuracy = self.get_accuracy(self.X_dev, self.Y_dev)

                if accuracy >= save_over:
                    self.save_weights()

                print(f"Iteration {i}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")

    def calculate_loss(self):
        m = self.Y_train.shape[0]
        log_probs = np.log(self.A2 + 1e-8)  # Add small epsilon to avoid log(0)
        loss = -np.sum(self.Y_train * log_probs) / m
        return loss

    def get_accuracy(self, X, Y):
        Z1 = X.dot(self.W1) + self.b1
        A1 = ReLU(Z1)
        Z2 = A1.dot(self.W2) + self.b2
        A2 = softmax(Z2)
        predictions = np.argmax(A2, axis=1)
        return np.mean(predictions == np.argmax(Y, axis=1))

    def predict(self, X, r=0):
        X = X / 255.0  # Normalize pixel values
        plot_time_series(X[r], "input")
        Z1 = X.dot(self.W1) + self.b1
        plot_time_series(Z1[r], "layer 1")
        A1 = ReLU(Z1)
        plot_time_series(A1[r], "NORM layer 1")
        Z2 = A1.dot(self.W2) + self.b2
        plot_time_series(Z2[r], "layer 2")
        A2 = softmax(Z2)
        plot_time_series(A2[r], "NORM final")
        return np.argmax(A2, axis=1)

    def save_weights(self, overwrite=True):
        weights_dict = {
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist()
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
        if os.path.exists(self.json_file):
            with open(self.json_file, 'r') as f:
                data = json.load(f)

            # Get the last set of weights from the loaded data
            weights_dict = data[-1]

            # Convert lists back to numpy arrays
            self.W1 = np.array(weights_dict["W1"])
            self.b1 = np.array(weights_dict["b1"])
            self.W2 = np.array(weights_dict["W2"])
            self.b2 = np.array(weights_dict["b2"])
        else:
            raise FileNotFoundError(f"File '{self.json_file}' not found.")

    def graph_params(self, param_series=None):
        # if param_series is None:
        #     param_series = self.b1_over_time
        # elif param_series == 2:
        #     param_series = self.b2_over_time
        #
        # observations = param_series
        # one_item = param_series[0]
        # list_of_vectors = [list(vec[0]) for vec in observations]
        # cleaned = []
        # for vec in list_of_vectors:
        #     ls = []
        #     for num in vec:
        #         ls.append(int(num*1000))
        #     cleaned.append(ls)
        # # cleaned is a list of lists of ints
        # vector_length = len(cleaned[0])
        # iterations = range(len(cleaned))
        #
        # print(cleaned[:2])



        # if they ask for more  values than we have, we give em all we got
        # indices = list(range(vector_length))


        #
        plt.figure(figsize=(12, 6))
        # for each dimension, plot the iterations vs the DIMth dimension of the vector
        # for i in indices:
        #     x = [x[i] for x in cleaned]
        #     plt.plot(iterations, x, label=f'Dimension {i}')

        plt.plot(range(len(self.b11_over_time)), self.b11_over_time)


        plt.xlabel('Iterations')
        plt.ylabel('Bias Value')
        # plt.title(f'Bias Vector Evolution During Training (Showing {values} dimensions)')
        # plt.legend()
        plt.show()

def plot_time_series(data, name):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(data)), data)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"{name}")
    plt.grid(True, axis='y')
    plt.show()


if __name__ == '__main__':

    # Usage
    nn = NeuralNetwork_MNIST(use_pretrained=True)
    # nn.gradient_descent(iterations=500, alpha=0.27)

    # After training, you can make predictions
    test_data = pd.read_csv('mnist_test.csv').values
    X_test = test_data[873:927, 1:]
    answers = test_data[873:927, 0]

    r = random.choice(np.arange(54))

    plot_image_from_vector(X_test[r])

    print(answers[r])
    predictions = nn.predict(X_test, r=r)

    print(predictions[r])

    h = zip(answers, predictions)

    x = [str(int(f[0]))+'-'+str(int(f[1])) for f in h]

    print(x)


    # nn.graph_params()
