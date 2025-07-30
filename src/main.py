from utils import load_gzip_to_data, plot_image
from layer import Layer

if __name__ == "__main__":
    training_data, validation_data, test_data = load_gzip_to_data('data/mnist.pkl.gz')
    images, labels = training_data
    image0 = images[17]
    label0 = labels[17]
    plot_image(image0, label0)

    input_layer = Layer(784)
    layer2 = Layer(16)
    layer3 = Layer(16)
    output_layer = Layer(10)

    input_layer.connect_to(layer2)
    layer2.connect_to(layer3)
    layer3.connect_to(output_layer)

    input_layer.load_data(image0)

    input_layer.forward()
    output_layer.print_state()

