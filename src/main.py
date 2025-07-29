from utils import load_gzip_to_data, plot_image
from layer import Layer

if __name__ == "__main__":
    training_data, validation_data, test_data = load_gzip_to_data('data/mnist.pkl.gz')
    images, labels = training_data
    image0 = images[0]
    label0 = labels[0]
    #plot_image(image0, label0)

    layer = Layer(784)
    layer.feed_data(image0)

    layer2 = Layer(128)
    layer.connect_to(layer2)

    layer3 = Layer(128)
    layer2.connect_to(layer3)

    last_layer = Layer(10)
    layer3.connect_to(last_layer)

    print("======== Layer 1 state ========")
    layer.feed_forward()
    layer.print_state()

    print("======== Layer 2 state ========")
    layer2.feed_forward()
    layer2.print_state()

    print("======== Layer 3 state ========")
    layer3.feed_forward()
    layer3.print_state()

    print("======== Last Layer state ========")
    last_layer.feed_forward()
    last_layer.print_state()

