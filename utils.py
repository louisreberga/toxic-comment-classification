from constants import EMBEDDINGS, VOCAB_SIZE
import models
import numpy as np
from keras.metrics import AUC
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score


def train_model(model, X_train, y_train, epochs, batch_size):
    early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=1, verbose=1)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[AUC()])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
              callbacks=[early_stopping], verbose=1)

    return model


def define_model(model_name, emb_matrix, emb_dim):
    if model_name == 'CNN':
        return models.cnn(emb_matrix, emb_dim)
    elif model_name == 'LSTM':
        return models.lstm(emb_matrix, emb_dim)
    elif model_name == 'bidirectional LSTM':
        return models.bidirectional_lstm(emb_matrix, emb_dim)
    elif model_name == 'GRU':
        return models.gru(emb_matrix, emb_dim)
    elif model_name == 'bidirectional GRU':
        return models.bidirectional_gru(emb_matrix, emb_dim)


def create_embedding_matrix(tokenizer, emb_name, emb_dim):
    def get_coefficients(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings = dict(get_coefficients(*o.strip().split()) for o in open(f'embeddings/{EMBEDDINGS[emb_name]}'))
    word_index = tokenizer.word_index
    all_emb = np.stack(embeddings.values())
    emb_mean, emb_std = all_emb.mean(), all_emb.std()
    emb_matrix = np.random.normal(emb_mean, emb_std, (VOCAB_SIZE, emb_dim))

    for word, i in word_index.items():
        emb_vector = embeddings.get(word)
        if emb_vector is not None:
            emb_matrix[i] = emb_vector

    return emb_matrix


def calculate_score(model, X_test, y_test, model_name, emb_name):
    y_pred = model.predict(X_test, batch_size=1024, verbose=1)
    score = roc_auc_score(y_test, y_pred)
    print(f'\nThe mean column-wise ROC AUC score for {model_name} with {emb_name} is {score:.5} \n\n\n')
