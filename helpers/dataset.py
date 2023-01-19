
import os


def generate_dataset():

    # take all images from the "raw" folder
    # each image have this pattern name: "<country>_<coin_value>_<specificity>_<id>.png"
    # country is a one or two letter code
    # coin_value is a string (e.g. "2euro", "1cent", "50cent")
    # specificity is a string (e.g. "eur", "com")

    # each coin of each country have several images (id from 0 to 9)
    # we need to split the images in train and test set
    # 80% of the images will be used for training, 20% for testing

    # so we just need to move into the "train" and "test" folder the images

    # we need to create the "train" and "test" folder if they don't exist

    FOLDER = "raw"
    TRAIN_FOLDER = "train"
    TEST_FOLDER = "test"

    # get all images
    images = os.listdir(FOLDER)

    # create the train and test folder if they don't exist
    if not os.path.exists(TRAIN_FOLDER):
        os.mkdir(TRAIN_FOLDER)
    if not os.path.exists(TEST_FOLDER):
        os.mkdir(TEST_FOLDER)

    # for each image
    for image in images:
        # get the country
        country = image.split("_")[0]

        # move the image in the train or test folder
        if int(image.split("_")[-1].split(".")[0]) < 8:
            os.rename(os.path.join(FOLDER, image),
                      os.path.join(TRAIN_FOLDER, image))
        else:
            os.rename(os.path.join(FOLDER, image),
                      os.path.join(TEST_FOLDER, image))
