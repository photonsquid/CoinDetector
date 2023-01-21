# Import standard dependencies
import os
import random

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.metrics import Precision, Recall

from src.dataset import create_pairs
from src.models import make_embedding, make_siamese_model

KERNEL_LOCATION = "local"                # TODO: auto-detect kernel location
LOG_LEVEL = "info"                       # TODO: handle log level
DATASET_SOURCE = "huggingface"
DATASET_NAME = "photonsquid/coins-euro"


if (DATASET_SOURCE == "huggingface"):
    from datasets.load import load_dataset as HF_load_dataset
    dataset = HF_load_dataset('photonsquid/coins-euro')
elif (DATASET_SOURCE == "local"):
    from src.helpers.load_data import load_data as local_load_data
    dataset = local_load_data("data/tests")
else:
    print("Invalid dataset source")

train_dataset = dataset["train"]

anchor_imgs, validation_imgs, labels = create_pairs(train_dataset)

# convert imgs and labels to tensors
anchor_imgs = tf.convert_to_tensor(anchor_imgs)
validation_imgs = tf.convert_to_tensor(validation_imgs)
labels = tf.convert_to_tensor(labels)

# create a tensorflow dataset
data = tf.data.Dataset.from_tensor_slices(
    (anchor_imgs, validation_imgs, labels))

for image in data:
    anchor = image[0].numpy()
    validation = image[1].numpy()
    label = "positive" if (image[2].numpy() == 1) else "negative"

    # show the images
    plt.subplot(1, 2, 1)
    plt.imshow(anchor)
    plt.axis('off')
    plt.title("anchor (" + str(image[0].shape) + ")")
    plt.subplot(1, 2, 2)
    plt.imshow(validation)
    plt.axis('off')
    plt.title(label + "(" + str(image[1].shape) + ")")
    plt.show()

    break


embedding = make_embedding()
siamese_model = make_siamese_model(embedding)

data = data.cache()
data = data.shuffle(buffer_size=10000)


# Training partition
# size of one batch
train_data = data.batch(16)

# prefetch data for faster training (prefetching means that the data is
# preprocessed (e.g. batched) while the model is training on the previous batch)
# 8 is the number of batches that will be prepared in advance
train_data = train_data.prefetch(8)

binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4)  # 0.0001

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)


@tf.function
def train_step(batch):

    # Record all of our operations
    with tf.GradientTape() as tape:
        # Get anchor and positive/negative image
        X = batch[:2]
        # Get label
        y = batch[2]

        # Forward pass
        yhat = siamese_model(X, training=True)
        # Calculate loss
        loss = binary_cross_loss(y, yhat)
    print(loss)

    # Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)

    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

    # Return loss
    return loss


def train(data, EPOCHS):
    # Loop through epochs
    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))

        # Creating a metric object
        r = Recall()
        p = Precision()

        # Loop through each batch
        for idx, batch in enumerate(data):
            # Run train step here
            loss = train_step(batch)
            yhat = siamese_model.predict(batch[:2])
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat)
            progbar.update(idx+1)
        print(loss.numpy(), r.result().numpy(), p.result().numpy())

        # Save checkpoints
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


EPOCHS = 50
train(train_data, EPOCHS)

test_dataset = dataset["test"]

test_anchor_imgs, test_validation_imgs, test_labels = create_pairs(
    test_dataset)
# convert imgs and labels to tensors
test_anchor_imgs = tf.convert_to_tensor(test_anchor_imgs)
test_validation_imgs = tf.convert_to_tensor(test_validation_imgs)
test_labels = tf.convert_to_tensor(test_labels)

# create a tensorflow dataset
test_data = tf.data.Dataset.from_tensor_slices(
    (test_anchor_imgs, test_validation_imgs, test_labels))

test_data = test_data.cache()
test_data = test_data.shuffle(buffer_size=10000)
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)
test_input, test_val, y_true = test_data.as_numpy_iterator().next()
y_hat = siamese_model.predict([test_input, test_val])
# Post processing the results
results = np.array([1 if prediction > 0.5 else 0 for prediction in y_hat])
# compute difference between predicted and true labels
diff = results - np.array(y_true)

# compute accuracy
accuracy = 1 - (np.count_nonzero(diff) / len(diff))

# save accuracy to file
with open("accuracy.txt", "w") as f:
    f.write(str(accuracy))


print("Accuracy: ", accuracy)
# save model
siamese_model.save("model")
# export model to file
tf.saved_model.save(siamese_model, "model")
