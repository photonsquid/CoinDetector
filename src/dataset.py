import random

import numpy as np
import tensorflow as tf


def preprocess(file_path):

    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image
    img = tf.io.decode_jpeg(byte_img)

    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100, 100))
    # Scale image to be between 0 and 1
    img = img / 255.0  # type: ignore

    # Return image
    return img

# we'll use the following function to create the pairs


def create_pairs(x, y):
    """Create positive and negative pairs from two arrays"""
    # create an empty list for the pairs
    pairs = []
    # create an empty list for the labels
    labels = []
    # create a list of unique classes
    classes = np.unique(y)
    # loop over the classes
    for c in classes:
        # find the indices of the images with the current class
        idx = np.where(y == c)[0]
        # loop over the indices
        for i in range(len(idx)):
            # get the current index
            z1, z2 = idx[i], idx[(i + 1) % len(idx)]
            # add the pair to the list of pairs
            pairs += [[x[z1], x[z2]]]
            # add the label to the list of labels
            inc = random.randrange(1, len(classes))
            dn = (c + inc) % len(classes)
            labels += [c == dn]
    # convert the pairs and labels to numpy arrays
    pairs = np.array(pairs)
    labels = np.array(labels)
    # return the pairs and labels
    return pairs, labels
