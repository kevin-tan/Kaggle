import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Dropout


# Digits Predictions, Pixels 28x28
def digitsPredict():
    img_rows, img_cols = 28, 28
    num_classes = 10

    def data_prep(raw):
        out_y = keras.utils.to_categorical(raw.label, num_classes)
        num_images = raw.shape[0]
        x_as_array = raw.values[:, 1:]
        x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)  # Channel 1 is since greyscale image
        out_x = x_shaped_array / 255  # To get value 0 to 1 for adam
        return out_x, out_y

    train_file = '../data/mnist/digit/train/train.csv'
    raw_data = pd.read_csv(train_file)

    X, y = data_prep(raw_data)

    # Start adding the layers to the network
    model = Sequential()
    # kernel_size is the convolution size
    model.add(Conv2D(filters=20, kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols, 1)))
    model.add(Conv2D(filters=20, kernel_size=(3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Configure the network
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

    # Train model
    model.fit(X, y, batch_size=128, epochs=2, validation_split=0.2)


# Data preparation
img_rows, img_cols = 28, 28
num_classes = 10


def prep_data(raw):
    # out_y prep
    # Making into a One-hot categorical (T or F)
    out_y = keras.utils.to_categorical(raw.label, num_classes)

    # out_x prep
    x = raw.values[:, 1:]
    num_images = raw.shape[0]  # Return a tuple with dimensions of dataframe
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    out_x = out_x / 255

    return out_x, out_y


fashion_file = '../data/mnist/clothing/train/fashion-mnist_train.csv'
data_raw = pd.read_csv(fashion_file)
X, y = prep_data(data_raw)

# Neural network setup
model = Sequential()
model.add(Conv2D(filters=12, kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(filters=12, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=12, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Configure network
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

# Training time
model.fit(X, y, batch_size=100, epochs=4, validation_split=0.2)
