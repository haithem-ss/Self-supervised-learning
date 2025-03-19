# Functions for data handling will be placed here

import os
import shutil
import tensorflow as tf
import requests
from zipfile import ZipFile


def filter_dataset(dataset_dir, selected_labels, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for label in selected_labels:
        src_label_dir = os.path.join(dataset_dir, label, "images")
        dest_label_dir = os.path.join(target_dir, label)
        os.makedirs(dest_label_dir, exist_ok=True)

        # Copy images for the selected label
        for img_file in os.listdir(src_label_dir):
            src_img_path = os.path.join(src_label_dir, img_file)
            dest_img_path = os.path.join(dest_label_dir, img_file)
            shutil.copy(src_img_path, dest_img_path)


def filter_validation_dataset(val_dir, selected_labels, target_dir):
    val_images_dir = os.path.join(val_dir, "images")
    val_annotations_path = os.path.join(val_dir, "val_annotations.txt")

    # Read the val_annotations.txt file
    with open(val_annotations_path, "r") as f:
        annotations = f.readlines()

    # Parse the annotations to get image-to-label mapping
    img_label_map = {}
    for line in annotations:
        parts = line.strip().split("\t")
        img_name, label = parts[0], parts[1]
        img_label_map[img_name] = label

    # Filter images for the selected labels
    os.makedirs(target_dir, exist_ok=True)
    for img_name, label in img_label_map.items():
        if label in selected_labels:
            src_img_path = os.path.join(val_images_dir, img_name)
            dest_label_dir = os.path.join(target_dir, label)
            os.makedirs(dest_label_dir, exist_ok=True)
            dest_img_path = os.path.join(dest_label_dir, img_name)
            shutil.copy(src_img_path, dest_img_path)


def download_and_unzip_dataset(url, output_dir, zip_filename="dataset.zip"):
    # Download the dataset
    response = requests.get(url, stream=True)
    with open(zip_filename, "wb") as file:
        for chunk in response.iter_content(chunk_size=128):
            file.write(chunk)

    # Unzip the dataset
    with ZipFile(zip_filename, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    # Remove the zip file
    os.remove(zip_filename)


def rotate_image(image, rotation_label):
    if rotation_label == 1:  # 90 degrees
        image = tf.image.rot90(image, k=1)
    elif rotation_label == 2:  # 180 degrees
        image = tf.image.rot90(image, k=2)
    elif rotation_label == 3:  # 270 degrees
        image = tf.image.rot90(image, k=3)
    return image


def preprocess_data(image):

    image = tf.image.resize(image, (64, 64))  # Resize to match Tiny ImageNet size
    image = tf.cast(image, tf.float32) / 255.0
    rotation_label = tf.random.uniform(
        shape=[], minval=0, maxval=4, dtype=tf.int32
    )  # Random rotation label
    rotated_image = rotate_image(image, rotation_label)
    return rotated_image, rotation_label
