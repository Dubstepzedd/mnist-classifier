import gzip
import pickle
from matplotlib import pyplot as plt
import numpy as np

def load_gzip_to_data(filename, encoding='latin1'):
    with gzip.open(filename, 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding=encoding)
    # Reshape the images in each dataset to (num_samples, 28, 28)
    def reshape_images(data):
        images, labels = data
        images = images.reshape(-1, 28, 28)
        return (images, labels)

    training_data = reshape_images(training_data)
    validation_data = reshape_images(validation_data)
    test_data = reshape_images(test_data)
    return (training_data, validation_data, test_data)


def evaluate_accuracy(predictions, labels):
    correct = np.sum(predictions == labels)
    total = len(labels)
    return correct / total


def plot_image(image, label):
    plt.imshow(image, cmap='gray')
    plt.title(f'Label: {label}')
    plt.axis('off')
    plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def softmax(x):
    exps = np.exp(x - np.max(x))  # subtract max for stability
    return exps / np.sum(exps)

def cross_entropy_loss(predictions, targets):
    # predictions and targets are vectors (batch size = 1 here)
    # Add a tiny epsilon to avoid log(0)
    epsilon = 1e-12
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    return -np.sum(targets * np.log(predictions))

def derivative_cross_entropy_softmax(logits, targets):
    # logits: raw output before softmax
    # targets: one-hot encoded true labels
    probs = softmax(logits)
    return probs - targets
