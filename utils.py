from constants import EMBEDDING, MODEL, EMBEDDINGS, VOCAB_SIZE, EMB_DIM, EPOCHS, BATCH_SIZE
import models
import numpy as np
from keras.metrics import AUC
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score


def train_model(model, X_train, y_train):
    early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=1, verbose=1)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[AUC()])
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1,
              callbacks=[early_stopping], verbose=1)

    return model


def define_model(emb_matrix):
    if MODEL == 'LSTM':
        return models.lstm(emb_matrix)
    elif MODEL == 'bidirectional LSTM':
        return models.bidirectional_lstm(emb_matrix)


def create_embedding_matrix(tokenizer):
    def get_coefficients(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings = dict(get_coefficients(*o.strip().split()) for o in open(f'embeddings/{EMBEDDINGS[EMBEDDING]}'))
    word_index = tokenizer.word_index
    all_emb = np.stack(embeddings.values())
    emb_mean, emb_std = all_emb.mean(), all_emb.std()
    emb_matrix = np.random.normal(emb_mean, emb_std, (VOCAB_SIZE, EMB_DIM))

    for word, i in word_index.items():
        emb_vector = embeddings.get(word)
        if emb_vector is not None:
            emb_matrix[i] = emb_vector

    return emb_matrix


def calculate_score(model, X_test, y_test):
    y_pred = model.predict(X_test, batch_size=1024, verbose=1)
    score = roc_auc_score(y_test, y_pred)
    print(f'\nThe mean column-wise ROC AUC score for {MODEL} with {EMBEDDING} is {score:.5}')
