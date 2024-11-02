import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
                 json_file='MNIST_W&B.json'):
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
            self.data_dev = self.data[:1000]
            self.Y_dev = one_hot(self.data_dev[:, 0])
            self.X_dev = self.data_dev[:, 1:] / 255.0  # Normalize pixel values to [0, 1]

            self.data_train = self.data[1000:]
            self.Y_train = one_hot(self.data_train[:, 0])
            self.X_train = self.data_train[:, 1:] / 255.0  # Normalize pixel values to [0, 1]

        # Initialize weights and biases
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.init_params()

        # Placeholders for intermediate values during forward pass
        self.Z1 = None
        self.A1 = None
        self.Z2 = None
        self.A2 = None

        # Placeholders for gradients
        self.dW1 = None
        self.db1 = None
        self.dW2 = None
        self.db2 = None

    def init_params(self):
        """
        Initialize random weights and biases.
        W1: (784, 100) - connects input layer (784 nodes) to hidden layer (100 nodes)
        b1: (1, 100) - bias for hidden layer
        W2: (100, 10) - connects hidden layer (100 nodes) to output layer (10 nodes)
        b2: (1, 10) - bias for output layer
        """
        if self.use_pretrained:
            return self.load_weights()

        # He initialization for weights
        self.W1 = np.random.randn(784, 100) / np.sqrt(784)
        self.b1 = np.zeros((1, 100))
        self.W2 = np.random.randn(100, 10) / np.sqrt(100)
        self.b2 = np.zeros((1, 10))

    def save_weights(self, overwrite=True):
        """
        Save weights and biases to a JSON file.
        """
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
        else:
            raise FileNotFoundError(f"File '{self.json_file}' not found.")

    def feed_forward_train(self):
        """
        Perform forward propagation on the training data.
        """
        # First layer
        self.Z1 = np.dot(self.X_train, self.W1) + self.b1  # Shape: (m, 100)
        self.A1 = ReLU(self.Z1)  # Shape: (m, 100)

        # Second (output) layer
        self.Z2 = np.dot(self.A1, self.W2) + self.b2  # Shape: (m, 10)
        self.A2 = softmax(self.Z2)  # Shape: (m, 10)

    def back_prop(self):
        """
        Perform backpropagation to compute gradients.
        """
        m = self.X_train.shape[0]

        # Output layer
        dZ2 = self.A2 - self.Y_train  # Shape: (m, 10)
        self.dW2 = (1 / m) * np.dot(self.A1.T, dZ2)  # Shape: (100, 10)
        self.db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)  # Shape: (1, 10)

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
        log_probs = np.log(self.A2 + 1e-8)  # Add small epsilon to avoid log(0)
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
        A2 = softmax(Z2)
        predictions = np.argmax(A2, axis=1)
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
        A2 = softmax(Z2)
        return np.argmax(A2, axis=1)

    def visualize_misclassified(self, num_samples=10):
        """
        Visualize misclassified images from the test set.

        :param num_samples: Number of misclassified samples to display
        """
        test_data = self.test_data.values
        X_test = test_data[:, 1:] / 255.0
        y_test = test_data[:, 0]

        predictions = self.predict(X_test)
        misclassified = np.where(predictions != y_test)[0]

        num_samples = min(num_samples, len(misclassified))
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        fig.suptitle("Misclassified Images", fontsize=16)

        for i, idx in enumerate(np.random.choice(misclassified, num_samples, replace=False)):
            ax = axes[i // 5, i % 5]
            ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
            ax.set_title(f"True: {y_test[idx]}, Pred: {predictions[idx]}")
            ax.axis('off')

        plt.tight_layout()
        # plt.show()
        plt.savefig("misclassified.png")

    def visualize_network_state(self, sample_index):
        """
        Visualize the network's internal state during forward propagation for a single sample.

        :param sample_index: Index of the sample to visualize
        """
        # Get a single sample
        X = self.test_data.iloc[sample_index, 1:].values.reshape(1, -1) / 255.0
        y_true = self.test_data.iloc[sample_index, 0]

        # Forward propagation
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = ReLU(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = softmax(Z2)
        prediction = np.argmax(A2)

        # Visualization
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle(f"Network State Visualization (True: {y_true}, Predicted: {prediction})", fontsize=16)

        # Input image
        ax1 = fig.add_subplot(231)
        ax1.imshow(X.reshape(28, 28), cmap='gray')
        ax1.set_title("Input Image")
        ax1.axis('off')

        # First layer weights
        ax2 = fig.add_subplot(232)
        im = ax2.imshow(self.W1, cmap='coolwarm', aspect='auto')
        ax2.set_title("First Layer Weights")
        plt.colorbar(im, ax=ax2)

        # First layer activations
        ax3 = fig.add_subplot(233)
        ax3.bar(range(100), A1[0])
        ax3.set_title("First Layer Activations")
        ax3.set_xlabel("Neuron")
        ax3.set_ylabel("Activation")

        # Second layer weights
        ax4 = fig.add_subplot(234)
        im = ax4.imshow(self.W2, cmap='coolwarm', aspect='auto')
        ax4.set_title("Second Layer Weights")
        plt.colorbar(im, ax=ax4)

        # Output probabilities
        ax5 = fig.add_subplot(235)
        ax5.bar(range(10), A2[0])
        ax5.set_title("Output Probabilities")
        ax5.set_xlabel("Digit")
        ax5.set_ylabel("Probability")
        ax5.set_xticks(range(10))

        # Add text annotations for probabilities
        for i, prob in enumerate(A2[0]):
            ax5.text(i, prob, f'{prob:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig("oneExampleSeenThrough.png")

    def enhanced_visualize_network_state(self, sample_index):
        """
            Visualize the network's internal state during forward propagation for a single sample.

            :param sample_index: Index of the sample to visualize
            """
        # Get a single sample
        X = self.test_data.iloc[sample_index, 1:].values.reshape(1, -1) / 255.0
        y_true = self.test_data.iloc[sample_index, 0]

        # Forward propagation
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = ReLU(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = softmax(Z2)
        prediction = np.argmax(A2)

        # Visualization
        fig = plt.figure(figsize=(20, 20))
        fig.suptitle(f"Network State Visualization\nTrue: {y_true}, Predicted: {prediction}", fontsize=24, y=0.95)

        # Input image
        ax1 = fig.add_subplot(331)
        ax1.imshow(X.reshape(28, 28), cmap='gray')
        ax1.set_title("Input Image", fontsize=18)
        ax1.axis('off')

        # Input vector as bar chart
        ax2 = fig.add_subplot(332)
        ax2.bar(range(784), X.flatten(), color='gray', edgecolor='black')
        ax2.set_title("Input Vector", fontsize=18)
        ax2.set_xlabel("Pixel Index", fontsize=14)
        ax2.set_ylabel("Pixel Value", fontsize=14)
        ax2.set_xlim(0, 783)

        # Network architecture diagram
        ax3 = fig.add_subplot(333)
        ax3.axis('off')
        ax3.text(0.5, 0.5, "Network Architecture\n\nInput (784) → Dense (100) → Dense (10)",
                 ha='center', va='center', fontsize=16, wrap=True)

        # First layer weights
        ax4 = fig.add_subplot(334)
        im4 = ax4.imshow(self.W1, cmap='hot', aspect='auto')
        ax4.set_title("First Layer Weights", fontsize=18)
        divider4 = make_axes_locatable(ax4)
        cax4 = divider4.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im4, cax=cax4)

        # First layer activations
        ax5 = fig.add_subplot(335)
        ax5.bar(range(100), A1[0], color='red', edgecolor='darkred')
        ax5.set_title("First Layer Activations", fontsize=18)
        ax5.set_xlabel("Neuron", fontsize=14)
        ax5.set_ylabel("Activation", fontsize=14)

        # Heatmap of first layer activations
        ax6 = fig.add_subplot(336)
        im6 = ax6.imshow(A1.reshape(10, 10), cmap='hot', aspect='auto')
        ax6.set_title("First Layer Activations Heatmap", fontsize=18)
        divider6 = make_axes_locatable(ax6)
        cax6 = divider6.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im6, cax=cax6)

        # Second layer weights
        ax7 = fig.add_subplot(337)
        im7 = ax7.imshow(self.W2, cmap='hot', aspect='auto')
        ax7.set_title("Second Layer Weights", fontsize=18)
        divider7 = make_axes_locatable(ax7)
        cax7 = divider7.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im7, cax=cax7)

        # Output probabilities
        ax8 = fig.add_subplot(338)
        bars = ax8.bar(range(10), A2[0], color='red', edgecolor='darkred')
        ax8.set_title("Output Probabilities", fontsize=18)
        ax8.set_xlabel("Digit", fontsize=14)
        ax8.set_ylabel("Probability", fontsize=14)
        ax8.set_xticks(range(10))

        # Add text annotations for probabilities
        for bar in bars:
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("enhanced_network_visualization.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

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
    # tools/NeuralNetworks/MNIST_NN/
    nn = NeuralNetwork_MNIST(use_pretrained=True, json_file='trained_784_100_10.json')
    #
    # n = np.array(n) * 255
    # ddd = pd.read_csv('mnist_test.csv')
    # arr = ddd.loc[10][1:]
    # print(n)
    # print(nn.predict(n))

    # Visualize misclassified images
    # nn.visualize_misclassified()
    #
    # # Visualize network state for a random sample
    # random_sample_index = np.random.randint(0, len(nn.test_data))
    # nn.visualize_network_state(random_sample_index)
    # nn.enhanced_visualize_network_state(random_sample_index)