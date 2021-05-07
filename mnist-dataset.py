from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

# Loading data from the mnist - dataset and splitting the data into train and test parts.
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

network = models.Sequential()

# Network topology
"""This part of the code consists on building two densely connected layers 
    in order to process the data taken from the images. With the softmax attribute
    meaning a Probability density function with the sum of its elements being 1.
    Where the element with the biggest probability being the output of the network."""

network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28, )))
network.add(layers.Dense(10, activation='softmax'))

# The compilation Step - ?
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Preprocessing phase, where the data is reshaped and rescaled into [0, 1] interval.
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# Categorical Encoding - ?
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Training the network
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# Testing the Network
test_loss, tes_acc = network.evaluate(test_images, test_labels)
print(f'test_acc: {tes_acc}')
