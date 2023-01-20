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
    #           specificity: <specificity>,
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

    # then we will return a list of tuples (anchor_image, validation_image, computed_label)

    # to be more efficient, we will create a dictionary with the following structure

    # {
    #   country: {
    #       value: {
    #           specificity: {
    #               id: <image>
    #           }
    #       }
    #   }
    # }

    # this will allow us to quickly find a positive image

    # create the dictionary
    images_dict = {}

    # loop through the dataset
    for image in dataset:
        # get the country, value, specificity and id
        country = image["label"]["country"]
        value = image["label"]["value"]
        specificity = image["label"]["specificity"]
        id = image["label"]["id"]

        # if the country is not in the dictionary, add it
        if country not in images_dict:
            images_dict[country] = {}

        # if the value is not in the dictionary, add it
        if value not in images_dict[country]:
            images_dict[country][value] = {}

        # if the specificity is not in the dictionary, add it
        if specificity not in images_dict[country][value]:
            images_dict[country][value][specificity] = {}

        # add the image to the dictionary
        images_dict[country][value][specificity][id] = image

    # create the list of tuples
    images_pairs = []

    # loop through the dataset
    for image in dataset:
        # get the country, value, specificity and id
        country = image["label"]["country"]
        value = image["label"]["value"]
        specificity = image["label"]["specificity"]
        id = image["label"]["id"]

        # create the anchor image
        anchor_image = image["image"]

        # randomly select a positive or negative image
        if random.random() > 0.5:
            # select a positive image
            # this image will have the same country, value, specificity but different id
            # get a random id
            random_id = random.choice(
                list(images_dict[country][value][specificity].keys())
            )
            # get the image
            validation_image = images_dict[country][value][specificity][random_id]["image"]
            # create the label
            computed_label = np.array([1, 1, 1, 1, 1])
        else:
            # select a negative image
            # get a random country
            random_country = random.choice(list(images_dict.keys()))
            # get a random value
            random_value = random.choice(
                list(images_dict[random_country].keys()))
            # get a random specificity
            random_specificity = random.choice(
                list(images_dict[random_country][random_value].keys())
            )
            # get a random id
            random_id = random.choice(
                list(images_dict[random_country][random_value]
                     [random_specificity].keys())
            )
            # get the image
            validation_image = images_dict[random_country][random_value][
                random_specificity
            ][random_id]["image"]
            # create the label
            computed_label = np.array([0, 0, 0, 0, 0])

        # add the tuple to the list
        images_pairs.append((anchor_image, validation_image, computed_label))

    return images_pairs
