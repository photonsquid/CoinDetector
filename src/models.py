from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D, subtract
from keras.models import Model

from src.l1_dist import L1Dist


def make_embedding():
    inp = Input(shape=(105, 105, 3), name='input_image')

    # First block
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2, 2), padding='same')(c1)

    # Second block
    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2, 2), padding='same')(c2)

    # Third block
    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2, 2), padding='same')(c3)

    # Final embedding block
    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=[inp], outputs=[d1], name='embedding')


def make_siamese_model(embedding):

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


def make_triplet_model(embedding):

    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(105, 105, 3))

    # Positive image in the network
    positive_image = Input(name='positive_img', shape=(105, 105, 3))

    # Negative image in the network
    negative_image = Input(name='negative_img', shape=(105, 105, 3))

    # Combine triplet distance components
    positive_distance = L1Dist()
    positive_distance._name = 'positive_distance'
    positive_distance = positive_distance(embedding(input_image),
                                            embedding(positive_image))
    
    negative_distance = L1Dist()
    negative_distance._name = 'negative_distance'
    negative_distance = negative_distance(embedding(input_image),
                                            embedding(negative_image))

    # Classification layer
    classifier = Dense(1, activation='sigmoid')(subtract([positive_distance, negative_distance]))
    
    return Model(inputs=[input_image, positive_image, negative_image],
                    outputs=classifier, name='TripletNetwork')  