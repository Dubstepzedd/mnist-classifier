from enum import Enum
from utils import load_gzip_to_data
from network import Network


class Mode(Enum):
    TRAIN = "train"
    TEST = "test"
    TEST_WITH_VISUALS = "test_with_visuals"


MODE = Mode.TEST  # Change as needed

if __name__ == "__main__":
    training_data, validation_data, test_data = load_gzip_to_data('data/mnist.pkl.gz')

    # Initialize network
    net = Network([784, 32, 32, 10])

    if MODE in (Mode.TEST, Mode.TEST_WITH_VISUALS):
        if net.load_weights():
            print("Loaded saved weights.")
        else:
            print("No saved weights found.")

    if MODE == Mode.TRAIN:
        images, labels = training_data
        batch_size = 32
        learning_rate = 0.2

        print(f"{len(images)} training samples loaded.")

        for batch_start in range(0, len(images), batch_size):
            batch_images = images[batch_start:batch_start + batch_size]
            batch_labels = labels[batch_start:batch_start + batch_size]

            avg_loss = net.train_batch(batch_images, batch_labels, learning_rate)
            print(f"Batch {batch_start} — Avg Loss: {avg_loss:.4f}")

        print("Training complete. Saving weights...")
        net.save_weights()
        print("Saved.")

    elif MODE in (Mode.TEST, Mode.TEST_WITH_VISUALS):
        test_images, test_labels = test_data
        net.test(test_images[:100], test_labels[:100], with_visuals=(MODE == Mode.TEST_WITH_VISUALS))

    else:
        raise ValueError(f"Unknown mode: {MODE}")
