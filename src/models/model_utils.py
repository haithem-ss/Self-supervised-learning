# Functions for model building and training will be placed here

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


def build_pretext_task_model(num_classes=4):
    # Preprocessing layers for data augmentation
    data_augmentation = tf.keras.Sequential(
        [
            layers.Rescaling(1.0 / 255),  # Rescale pixel values to [0, 1]
            layers.RandomFlip("horizontal"),  # Randomly flip images horizontally
            layers.RandomZoom(0.1),  # Randomly zoom images by 10%
            layers.RandomContrast(0.1),  # Randomly adjust contrast
        ]
    )
    # Load the MobileNetV2 model as a feature extractor
    base_model = MobileNetV2(include_top=False, weights=None, input_shape=(64, 64, 3))
    base_model.trainable = True  # Fine-tune the base model

    x = data_augmentation(base_model.input)
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(base_model.input, outputs)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def build_downstream_task_model(num_classes=2):
    model = tf.keras.models.load_model("../../models/pretext_task_model.keras")

    # Modify the last layer
    base_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    new_output = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="binary_output"
    )(base_model.output)

    model = tf.keras.Model(inputs=base_model.input, outputs=new_output)
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
