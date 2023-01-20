
import os

import cv2


def load_data(path: str):
    # load the data from the path
    # this will return a dictionary with the following structure
    # {
    #   "test": [
    #       {
    #           "image": <image>,
    #           "label": {
    #               "country": <string>,
    #               "value": <string>,
    #               "specificity": <string>,
    #               "id": <string>
    #           }
    #       },
    #       ...
    #   ],
    #   "train": [
    #       {
    #           "image": <image>,
    #           "label": {
    #               "country": <string>,
    #               "value": <string>,
    #               "specificity": <string>,
    #               "id": <string>
    #           }
    #       },
    #       ...
    #   ]
    # }

    # get all folder names in the path

    folders = os.listdir(path)

    dataset = {}

    for folder in folders:

        dataset[folder] = []

        # get all file names in the folder
        folder_path = os.path.join(path, folder)
        files = os.listdir(folder_path)

        for file in files:
            # compute from the file name the country, value, specificity and id
            # pattern: country_value_specificity_id.png or country_value_id.png (specificity is optional)
            # example: italy_1_cent_2000_1.png or italy_1_cent_1.png
            file_name = file.split(".")[0]
            file_name = file_name.split("_")
            country = file_name[0]
            value = file_name[1]
            id = file_name[-1]
            if len(file_name) == 5:
                specificity = file_name[2]
            else:
                specificity = ""

            # load the image from the file
            image = cv2.imread(os.path.join(path, folder, file))

            # add the image to the dataset
            dataset[folder].append({
                "image": image,
                "labels": {
                    "country": country,
                    "value": value,
                    "specificity": specificity,
                    "id": id
                }
            })

    return dataset


if __name__ == "__main__":
    a = load_data("data/tests")
    print(a)
