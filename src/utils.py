import gzip
import pickle
from matplotlib import pyplot as plt

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

def plot_image(image, label):
    plt.imshow(image, cmap='gray')
    plt.title(f'Label: {label}')
    plt.axis('off')
    plt.show()