import numpy as np
from keras.layers import Input, UpSampling2D, Dense, Flatten, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import *


def image_preprocess(data: list) -> list:
    """

    :param data: 1D list of images path
    :return: 2D list [np.array1,...]
    """

    image_vector_data = []

    for path_range in range(len(data)):
        img = load_img(data[path_range], target_size=(256, 256))
        img = np.array(img, dtype=np.float32) / 255 - 0.5
        img = np.expand_dims(img, axis=0)
        image_vector_data.append(img)

    return image_vector_data


class ImageBranch(object):

    def __init__(self):

        self.pixel = Input(shape=(256, 256, 3))

    def encoder(self):
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(self.pixel)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
        encoded = Flatten()(encoded)
        encoded = Dense(units=96, activation='softmax')(encoded)
        return encoded

    def decoder(self, latent_vector):

        print(latent_vector.shape)
        latent_vector = Reshape(target_shape=(250, 128, 3))(latent_vector)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(latent_vector)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        return decoded
