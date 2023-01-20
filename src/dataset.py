import os
import random


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

    # we will also, in the same time, create a list of all different countries, and a list of all different coin_values
    # beacuse each country has the same number of coins, with the same values

    # howevever, each coin has a different number of specificities

    # we will create the following lists

    # countries = [country1, country2, country3, ...]
    # coin_values = [value1, value2, value3, ...]

    # we will create a dictionary of all the different specificities for each coin

    # images = {
    #       <country_name>:[
    #          <coin_value>:{
    #               <specificitie_name>: [
    #                   {
    #                           image: <image>,
    #                           id: <id>
    #                   },
    #                   ...
    #               ]
    #           },
    #           ...
    #       ]
    #   }

    # firsr, iterate over the dataset and create the images dictionary, the countries and coin_values lists

    images = {}
    countries = []
    coin_values = []

    for image in dataset:

        # get the country, value, specificity and id of the image
        country = image["label"]["country"]
        value = image["label"]["value"]
        specificity = image["label"]["specificity"] if image["label"]["specificity"] != "" else "null"
        id = image["label"]["id"]

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
            images[country][value][specificity] = []

        # add the image to the images dictionary
        images[country][value][specificity]["id"] = id
        images[country][value][specificity)["image"] = image["image"]

    # now that we have the images dictionary, the countries and coin_values lists
    # we can create the pairs

    images_pairs = []

    for country in countries:
        for coin_value in coin_values:
            # get list of all specificities for the current coin
            coin_specificities = images[country]["coins"][coin_value]["specificities"]
            for coin_specificity in coin_specificities:
                for image in images[country]["coins"][coin_value]["specificities"][coin_specificity]["images"]:
                    # randmoly choose a positive or negative image
                    positive = random.choice([True, False])
                    validation_image = None
                    computed_label = None
                    if positive:
                        computed_label = [1, 1, 1, 1, 1]
                        # if positive, same country, same coin_value and same_specificity, but different id
                        # we need to randomly choose a different id
                        while True:
                            # get a random image from the same country, same coin_value and same_specificity
                            validation_image = random.choice(
                                images[country]["coins"][coin_value]["specificities"][coin_specificity]["images"])
                            # check if the id is different
                            if validation_image["id"] != image["id"]:
                                # if different, break the loop
                                break
                    else:
                        # if negative, at least one different attribute (country, coin_value, coin_specificity)
                        computed_label = [0, 0, 0, 0, 0]

                        while True:
                            # get a random country
                            random_country = random.choice(countries)
                            # get a random coin_value
                            random_coin_value = random.choice(coin_values)
                            # get a random coin_specificity
                            random_coin_specificity = random.choice(
                                coin_specificities)
                            # get a random image
                            validation_image = random.choice(
                                images[random_country]["coins"][random_coin_value]["specificities"][random_coin_specificity]["images"])
                            # check if the country is different
                            if random_country != country or random_coin_value != coin_value or random_coin_specificity != coin_specificity:
                                # if different, break the loop
                                break

                    # add the pair to the images_pairs list
                    images_pairs.append(
                        (image["image"], validation_image["image"], computed_label))

    return images_pairs


if __name__ == "__main__":
    from helpers.load_data import load_data

    # load the dataset
    dataset = load_data('data/tests')
    # create the pairs
    images_pairs = create_pairs(dataset['train'])
