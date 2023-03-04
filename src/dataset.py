import os
import random

import numpy as np

def create_pairs(dataset, size=(105, 105)):

    # input

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

    # output
    # images = {
    #       <country_name>:{
    #          <coin_value>:{
    #               <edition_name>: [
    #                   {
    #                           image: <image>,
    #                           variant: <id>
    #                   },
    #                   ...
    #               ]
    #           },
    #           ...
    #       ]
    #   }

    # first, iterate over the dataset and create the images dictionary, the countries and coin_values lists

    images = {}
    countries = []
    coin_values = []

    for image in dataset:

        # get the country, value, specificity and id of the image
        country = image["labels"]["country"]
        value = image["labels"]["value"]
        specificity = image["labels"]["edition"]
        id = image["labels"]["variant"]

        # if the country is not in the countries list, add it
        if country not in countries:
            countries.append(country)

        if value not in coin_values:
            coin_values.append(value)

        if country not in images:
            images[country] = {}

        # check if value is an attribute of the images[country] dictionary
        if value not in images[country]:
            images[country][value] = {}

        if specificity not in images[country][value]:
            images[country][value][specificity] = {}

        # add the image to the images dictionary
        images[country][value][specificity][id] = image["image"]

    # now that we have the images dictionary, the countries and coin_values lists
    # we can create the pairs

    anchor_imgs = []
    validation_imgs = []
    computed_labels = []
    anchor_imgs = []

    for country in countries:
        for coin_value in coin_values:
            # get list of all specificities for the current coin
            coin_specificities = images[country][coin_value]
            for coin_specificity in coin_specificities:
                for image_id in images[country][coin_value][coin_specificity]:
                    # randomly choose a positive or negative image
                    list_of_images = list(
                        images[country][coin_value][coin_specificity])
                    positive = random.choice([True, False])
                    validation_image = None
                    computed_label = None
                    if positive:
                        computed_label = 1
                        # if positive, same country, same coin_value and same_specificity, but different id
                        # we need to randomly choose a different id
                        if len(list_of_images) == 1:
                            validation_image = images[country][coin_value][coin_specificity][image_id]
                        else:
                            while True:
                                # get a random image from the same country, same coin_value and same_specificity
                                validation_image_id = random.choice(
                                    list_of_images)
                                # check if the id is different
                                if validation_image_id != image_id:
                                    # if different, break the loop
                                    break
                            validation_image = images[country][coin_value][coin_specificity][validation_image_id]
                    else:
                        # if negative, at least one different attribute (country, coin_value, coin_specificity)
                        computed_label = 0

                        while True:
                            # get a random country
                            random_country = random.choice(countries)
                            # get a random coin_value
                            random_coin_value = random.choice(coin_values)
                            # get a random coin_specificity of the random coin_value of the random country
                            random_coin_specificity = random.choice(
                                list(images[random_country][random_coin_value]))

                            # get a random image from the random country, random coin_value and random coin_specificity
                            validation_image_id = random.choice(
                                list(images[random_country][random_coin_value][random_coin_specificity]))

                            # check if the country is different
                            if random_country != country or random_coin_value != coin_value or random_coin_specificity != coin_specificity:
                                # if different, break the loop
                                break
                        validation_image = images[random_country][random_coin_value][random_coin_specificity][validation_image_id]
                    anchor_image = images[country][coin_value][coin_specificity][image_id]

                    # remove the alpha channel
                    anchor_image = anchor_image.convert("RGB")
                    validation_image = validation_image.convert("RGB")

                    # resize the images
                    anchor_image = anchor_image.resize(size)
                    validation_image = validation_image.resize(size)

                    # convert the images to numpy arrays
                    anchor_image = np.array(anchor_image)
                    validation_image = np.array(validation_image)

                    # scale down the images
                    anchor_image = anchor_image / 255
                    validation_image = validation_image / 255

                    # add the pair to the images_pairs list
                    anchor_imgs.append(anchor_image)
                    validation_imgs.append(validation_image)
                    computed_labels.append(computed_label)
                    anchor_imgs.append(anchor_image)
                    validation_imgs.append(validation_image)
                    computed_labels.append(computed_label)

    return anchor_imgs, validation_imgs, computed_labels


def create_triplets(dataset, size=(105, 105)):

    images = {}
    countries = []
    coin_values = []

    for image in dataset:

        # get the country, value, specificity and id of the image
        country = image["labels"]["country"]
        value = image["labels"]["value"]
        specificity = image["labels"]["edition"]
        id = image["labels"]["variant"]

        # if the country is not in the countries list, add it
        if country not in countries:
            countries.append(country)

        if value not in coin_values:
            coin_values.append(value)

        if country not in images:
            images[country] = {}

        # check if value is an attribute of the images[country] dictionary
        if value not in images[country]:
            images[country][value] = {}

        if specificity not in images[country][value]:
            images[country][value][specificity] = {}

        # add the image to the images dictionary
        images[country][value][specificity][id] = image["image"]

    # now that we have the images dictionary, the countries and coin_values lists
    # we can create the pairs

    anchor_imgs = []
    positive_imgs = []
    negative_imgs = []

    for country in countries:
        for coin_value in coin_values:
            # get list of all specificities for the current coin
            coin_specificities = images[country][coin_value]
            for coin_specificity in coin_specificities:
                # Create a list of all possible combinations of images for the current coin
                
                
                for image_id in images[country][coin_value][coin_specificity]:
                    # randomly choose a positive and a negative image
                    list_of_images = list(
                        images[country][coin_value][coin_specificity])
                    positive_img = None
                    negative_img = None

                    # if positive, same country, same coin_value and same_specificity, but different id
                    # we need to randomly choose a different id
                    if len(list_of_images) == 2:
                        # if there are only two images, the positive image is the one with the different id
                        positive_img = list_of_images[1] # wait
                        negative_img = list_of_images[0]
                        validation_image = images[country][coin_value][coin_specificity][image_id]
                    else:
                        while True:
                            # get a random image from the same country, same coin_value and same_specificity
                            validation_image_id = random.choice(
                                list_of_images)
                            # check if the id is different
                            if validation_image_id != image_id:
                                # if different, break the loop
                                break
                        validation_image = images[country][coin_value][coin_specificity][validation_image_id]
                    else:
                        # if negative, at least one different attribute (country, coin_value, coin_specificity)
                        computed_label = 0

                        while True:
                            # get a random country
                            random_country = random.choice(countries)
                            # get a random coin_value
                            random_coin_value = random.choice(coin_values)
                            # get a random coin_specificity of the random coin_value of the random country
                            random_coin_specificity = random.choice(
                                list(images[random_country][random_coin_value]))

                            # get a random image from the random country, random coin_value and random coin_specificity
                            validation_image_id = random.choice(
                                list(images[random_country][random_coin_value][random_coin_specificity]))

                            # check if the country is different
                            if random_country != country or random_coin_value != coin_value or random_coin_specificity != coin_specificity:
                                # if different, break the loop
                                break
                        validation_image = images[random_country][random_coin_value][random_coin_specificity][validation_image_id]
                    anchor_image = images[country][coin_value][coin_specificity][image_id]

                    # remove the alpha channel
                    anchor_image = anchor_image.convert("RGB")
                    validation_image = validation_image.convert("RGB")

                    # resize the images
                    anchor_image = anchor_image.resize(size)
                    validation_image = validation_image.resize(size)

                    # convert the images to numpy arrays
                    anchor_image = np.array(anchor_image)
                    validation_image = np.array(validation_image)

                    # scale down the images
                    anchor_image = anchor_image / 255
                    validation_image = validation_image / 255

                    # add the pair to the images_pairs list
                    anchor_imgs.append(anchor_image)
                    validation_imgs.append(validation_image)
                    computed_labels.append(computed_label)
                    anchor_imgs.append(anchor_image)
                    validation_imgs.append(validation_image)
                    computed_labels.append(computed_label)

    return anchor_imgs, validation_imgs, computed_labels


if __name__ == "__main__":
    from helpers.load_data import load_data
    testing = False
    if testing == True:
        # load the dataset
        dataset = load_data('data/tests')
        pass
    else:
        from datasets.load import load_dataset
        dataset = load_dataset('photonsquid/coins-euro')
    # create the pairs
    anchor_imgs, validation_imgs, computed_labels = create_triplets(
        dataset['test'])
