from keras import Input
from keras.layers import Bidirectional, Dense, LSTM, RepeatVector, TimeDistributed, Dropout, Flatten
from keras.layers import Embedding
from keras.layers.advanced_activations import ELU
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from numpy import asarray
from numpy import zeros


def text_preprocess():
    # load text file into a list | sentences

    docs = ['Well Done', 'Dummy Text']

    t = Tokenizer()
    t.fit_on_texts(docs)
    vocab_size = len(t.word_index) + 1
    # integer encode the documents
    encoded_docs = t.texts_to_sequences(docs)
    # pad documents to a max length of 4 words
    max_length = 4
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    # load the whole embedding into memory
    embeddings_index = dict()
    f = open('./glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    # create a weight matrix for words in training docs
    embedding_matrix = zeros((vocab_size, 100))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


class TextBranch(object):

    def __init__(self, embedding_matrix):
        self.NB_WORDS = 5  # vocab size
        self.max_len = 1000
        self.latent_dim = 64  # encoding part flat size
        self.glove_embedding_matrix = embedding_matrix
        self.intermediate_dim = 96
        self.emb_dim = 100
        self.char = Input(batch_shape=(None, self.max_len))

    """
    def encoder(self):
        print(self.glove_embedding_matrix.shape)
        act = ELU()

        x = self.char
        x_embed = Embedding(self.NB_WORDS, self.emb_dim, weights=[self.glove_embedding_matrix],
                            input_length=self.max_len, trainable=True)(x)
        h = Bidirectional(LSTM(self.intermediate_dim, return_sequences=False, recurrent_dropout=0.2),
                          merge_mode='concat')(
            x_embed)
        # h = Bidirectional(LSTM(intermediate_dim, return_sequences=False), merge_mode='concat')(h)
        # h = Flatten()(h)
        h = Dropout(0.3)(h)
        h = Dense(self.intermediate_dim, activation='linear')(h)
        h = act(h)
        h = Dropout(0.2)(h)
        # encoded = Flatten()(h)
        encoder_ = Dense(128)(h)  # encoded = Dense(units=128, activation='softmax')(encoded)

        return encoder_

    def decoder(self, latent_vector):
        print("Latent vector shape is {}".format(latent_vector.shape))

        repeated_context = RepeatVector(self.max_len)
        decoder_h = LSTM(self.intermediate_dim, return_sequences=True, recurrent_dropout=0.2)
        decoder_mean = TimeDistributed(
            Dense(self.NB_WORDS, activation='linear'))  # softmax is applied in the seq2seqloss by tf
        h_decoded = decoder_h(repeated_context(latent_vector))
        decoder_ = decoder_mean(h_decoded)

        return decoder_
 
 """

    def encoder(self):
        print(f"max len is {self.max_len}")
        inputs = self.char
        embedding = Embedding(self.NB_WORDS, self.emb_dim, weights=[self.glove_embedding_matrix],
                              input_length=self.max_len, trainable=True)(inputs)
        encoded_h1 = Dense(256, activation='relu')(embedding)
        encoded_h2 = Dense(128, activation='relu')(encoded_h1)
        latent = Dense(96, activation='relu')(encoded_h2)
        return latent

    def decoder(self, latent_vector):

        decoder_h1 = Dense(4, activation='tanh')(latent_vector)
        decoder_h2 = Dense(8, activation='tanh')(decoder_h1)
        decoder_h3 = Dense(16, activation='tanh')(decoder_h2)
        decoder_h4 = Dense(32, activation='tanh')(decoder_h3)
        decoder_h5 = Dense(64, activation='tanh')(decoder_h4)

        output = Dense(self.max_len, activation='tanh')(decoder_h5)

        return output


# test code


