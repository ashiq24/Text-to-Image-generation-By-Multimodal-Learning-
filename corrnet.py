from keras.models import Model
from keras.layers import Input, UpSampling2D, Add
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import numpy as np

class IMAGE(object):

    def __init__(self):
        self.pixel = Input(shape=(256, 256, 3))

    def encoder(self):
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(self.pixel)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
        print(encoded)
        return encoded  # returns 1D tensor

    #@staticmethod
    def decoder(self,encoded):
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        return decoded


class TEXT(object):
    pass


class CORRELATIONALNN(object):
    def __init__(self):
        pass

    @staticmethod
    def build_model():
        # encoder
        IMAGE_CHANNEL1 = IMAGE()
        IMAGE_CHANNEL2 = IMAGE()
        encoder_channel_1  = IMAGE_CHANNEL1.encoder()
        encoder_channel_2 =  IMAGE_CHANNEL2.encoder()
        latent_vector = Add()([encoder_channel_1, encoder_channel_2])

        # decoder
        decoder_channel_1 = IMAGE_CHANNEL1.decoder(latent_vector)
        decoder_channel_2 = IMAGE_CHANNEL2.decoder(latent_vector)
        # build model

        model = Model(inputs=[IMAGE_CHANNEL1.pixel, IMAGE_CHANNEL2.pixel], outputs=[decoder_channel_1, decoder_channel_2])
        print(model.summary())


corrnn = CORRELATIONALNN()
corrnn.build_model()

