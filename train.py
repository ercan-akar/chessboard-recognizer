import os
from glob import glob
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

RATIO = 0.8                 # ratio of training vs. test data
LABELS = 'RNBQKPrnbqkp1'    # 13 labels for possible square contents

def image_data(image_path) -> tf.image:
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [32, 32])

def create_model() -> models.Sequential:
    """ Convolutional neural network for image classification. Same architecture as:
        https://www.tensorflow.org/tutorials/images/cnn
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(LABELS), activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def get_dataset():
    all_paths = np.array(glob("train_tiles/*/*.png"))
    np.random.seed(1)
    np.random.shuffle(all_paths)

    divider = int(len(all_paths) * RATIO)
    train_paths = all_paths[:divider]
    test_paths = all_paths[divider:]

    # TODO why does a list comprehension with np.array freeze??
    train_images = []
    train_labels = []
    for image_path in train_paths:
        piece_type = image_path[-5]
        assert(piece_type in LABELS)
        train_images.append(np.array(image_data(image_path)))
        train_labels.append(LABELS.index(piece_type))
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    print("Loaded {} training images and labels".format(len(train_paths)))

    test_images = []
    test_labels = []
    for image_path in test_paths:
        piece_type = image_path[-5]
        assert(piece_type in LABELS)
        test_images.append(np.array(image_data(image_path)))
        test_labels.append(LABELS.index(piece_type))
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    print("Loaded {} test images and labels".format(len(test_paths)))
    return ((train_images, train_labels), (test_images, test_labels))


if __name__ == '__main__':
    print('Tensorflow {}'.format(tf.version.VERSION))

    (train_images, train_labels), (test_images, test_labels) = get_dataset()
    model = create_model()
    model.fit(train_images, train_labels, epochs=10,
              validation_data=(test_images, test_labels))

    print("Evaluating model on test data:")
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)