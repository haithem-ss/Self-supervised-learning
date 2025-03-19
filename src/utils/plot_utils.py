# Functions for plotting will be placed here

import matplotlib.pyplot as plt
import tensorflow as tf


def plot_training_history(history):
    """Plot training and validation loss and accuracy."""
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Train and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_predictions(model, val_ds, num_images_to_plot=10):
    """Plot a set of validation images with their predicted labels."""
    val_images, val_labels = next(iter(val_ds))
    y_pred = model.predict(val_images)
    y_pred_classes = tf.argmax(y_pred, axis=1)

    plt.figure(figsize=(10, 5))
    for i in range(num_images_to_plot):
        plt.subplot(2, 5, i + 1)
        plt.imshow(val_images[i].numpy(), vmin=0, vmax=1)
        plt.title(f"Label: {y_pred_classes[i]}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
