from text_embedding import text_preprocess, TextBranch
from image_embedding import image_preprocess, ImageBranch
from keras.layers import Add
from keras.models import Model
import os


class CorrelationalNeuralNet(object):

    def __init__(self):
        pass

    def build_model(self):

        # preprocess data | Image | Text | From Dir ./dataset | /images | /captions

        # Preprocess Image Dataset | we give path list of all images and returns an 2D array of shape
        # total images * x * y * color_channel(3 in this case) | x,y = 256, 256 in this case
        # [ [image1(256,256,3)], [image2(256,256,3)] ]

        image_path_list = ['./dataset/images/{path}'.format(path=path)
                           for path in os.listdir('./dataset/images')]  # full path
        print("Total Images in the dataset is {}".format(len(image_path_list)))
        image_vector_data = image_preprocess(image_path_list)

        # Preprocess Text Dataset
        embedding_matrix, vocab_size = text_preprocess(["a boy is standing"])

        pixel_branch = ImageBranch()
        char_branch = TextBranch(embedding_matrix=embedding_matrix, vocab_size=vocab_size)

        # encoder
        pixel_encoder, char_encoder = pixel_branch.encoder(), char_branch.encoder()
        # latent space
        latent_space = Add()([pixel_encoder, char_encoder])
        # decoder
        pixel_decoder, char_decoder = pixel_branch.decoder(latent_space), char_branch.decoder(latent_space)

        # build Model

        corrnet = Model(inputs=[pixel_branch.pixel, char_branch.char],
                        outputs=[pixel_decoder, char_decoder]
                        )
        print(corrnet.summary())

        corrnet.compile(optimizer='adam', loss='binary_crossentropy')
        corrnet.fit(x=[image_vector_data, embedding_matrix], y=[image_vector_data, embedding_matrix],
                    batch_size=100,
                    epochs=90,
                    validation_data=(None, None)
                    )




CRNet = CorrelationalNeuralNet()
CRNet.build_model()



