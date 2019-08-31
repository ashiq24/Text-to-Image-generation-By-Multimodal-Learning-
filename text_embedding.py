from keras import Input
from keras.layers import Dense
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from numpy import asarray
from numpy import zeros


def text_preprocess():
    # load text file into a list | sentences

    docs = ['Well Done', 'Dummy Text']

    t = Tokenizer()
    t.fit_on_texts(docs)

    vocab_size = len(t.word_index) + 1
    print(vocab_size)
    # integer encode the documents
    # encoded_docs = t.texts_to_sequences(docs)
    # # pad documents to a max length of 4 words
    # max_length = 4
    # padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
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

    def __init__(self, embedding_matrix, vocab_size):
        self.NB_WORDS = vocab_size  # vocab size
        self.max_len = 1000
        self.glove_embedding_matrix = embedding_matrix
        self.emb_dim = 100
        self.char = Input(batch_shape=(None, self.max_len))

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


