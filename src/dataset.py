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


def create_pairs(dataset):

    # dataset is a list with the following structure

    # dataset = [
    #   {
    #       image: <image>,
    #       label:{
    #           country: <country>,
    #           value: <value>,
    #           spceificity: <specificity>,
    #           id: <id>,
    #       }
    #   },
    #   ...
    # ]

    # for each image in the dataset we need to create a pair, a pair is a
    # tuple of 2 images and one label (anchor_image, validation_image, computed_label)
    # the anchor image is the image we are trying to validate
    # the validation image is either a positive or negative image
    # the computed_label = 0,0,0,0,0 (negative) or 1,1,1,1,1 (positive)

    # a positive image is an image of the same country, same value, same specificity but different id
    # a negative image is an image of at least one diffrent attribute (country, value, specificity)

    # we need to randomly select a postive or negative image for each image in the dataset

    # then we will return
    # (anchor, positive, computed_label) or (anchor, negative, computed_label)
    # with computed_label = 0,0,0,0,0 (negative) or 1,1,1,1,1 (positive)

    #

    return (anchor, validaion, label)
