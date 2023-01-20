from keras.layers import Dense, Input
from keras.models import Model

from src.l1_dist import L1Dist


def make_siamese_model():

    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(105, 105, 3))

    # Validation image in the network
    validation_image = Input(name='validation_img', shape=(105, 105, 3))

    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image),
                              embedding(validation_image))

    # Classification layer
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')
