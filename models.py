from constants import VOCAB_SIZE, MAX_LENGTH
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM, GlobalMaxPool1D, Dropout, Bidirectional


def cnn(emb_matrix, emb_dim):
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, emb_dim, weights=[emb_matrix], input_length=MAX_LENGTH, trainable=False))
    # TODO

    return model


def lstm(emb_matrix, emb_dim):
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, emb_dim, weights=[emb_matrix], input_length=MAX_LENGTH, trainable=False))
    model.add(LSTM(60, return_sequences=True))
    model.add(GlobalMaxPool1D())
    model.add(Dropout(0.1))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(6, activation="sigmoid"))

    return model


def bidirectional_lstm(emb_matrix, emb_dim):
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, emb_dim, weights=[emb_matrix], input_length=MAX_LENGTH, trainable=False))
    model.add(Bidirectional(LSTM(60, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)))
    model.add(GlobalMaxPool1D())
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(6, activation="sigmoid"))

    return model


def gru(emb_matrix, emb_dim):
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, emb_dim, weights=[emb_matrix], input_length=MAX_LENGTH, trainable=False))
    # TODO

    return model


def bidirectional_gru(emb_matrix, emb_dim):
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, emb_dim, weights=[emb_matrix], input_length=MAX_LENGTH, trainable=False))
    # TODO

    return model

