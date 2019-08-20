from text_embedding import text_preprocess, TextBranch
from image_embedding import image_preprocess, ImageBranch
from keras.layers import Add
from keras.models import Model


class CorrelationalNeuralNet(object):

    def __init__(self):
        pass

    def build_model(self):

        embedding_matrix = text_preprocess()

        pixel_branch = ImageBranch()
        char_branch = TextBranch(embedding_matrix=embedding_matrix)  # embedding parameter as constructor

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


CRNet = CorrelationalNeuralNet()
CRNet.build_model()



