import pandas as pd
import numpy as np
import re

from nltk.corpus import stopwords
from gensim.models import KeyedVectors

from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM, GlobalMaxPool1D, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

MAX_LENGTH = 400
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
EMBEDDINGS = {
    "GloVe 6B 50D": "glove.6B.50d.txt",
    "GloVe 6B 100D": "glove.6B.100d.txt",
    "GloVe 6B 200D": "glove.6B.200d.txt",
    "GloVe 6B 300D": "glove.6B.300d.txt",
    "Word2vec Google News 300D": "GoogleNews-vectors-negative300.bin"
}


def main():
    train_df = pd.read_csv('data/train.csv.zip')
    train_df['comment_text'] = train_df['comment_text'].apply(clean_comment)
    list_comment_train = list(train_df['comment_text'])

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(list_comment_train)
    encoded_comment_train = tokenizer.texts_to_sequences(list_comment_train)

    vocab_size = len(tokenizer.word_index) + 1
    X_train = pad_sequences(encoded_comment_train, maxlen=MAX_LENGTH, padding='post')

    test_df = pd.read_csv('data/test.csv.zip')
    test_df['comment_text'] = train_df['comment_text'].apply(clean_comment)
    list_comment_test = list(test_df['comment_text'])

    encoded_comment_test = tokenizer.texts_to_sequences(list_comment_test)
    X_test = pad_sequences(encoded_comment_test, maxlen=MAX_LENGTH, padding='post')

    word2vec_model: KeyedVectors = KeyedVectors.load_word2vec_format('embeddings/GoogleNews-vectors-negative300.bin',
                                                                     binary=True)
    embedding_dim = word2vec_model[0].size

    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    words = tokenizer.word_index

    for word in words:
        index = words[word]

        try:
            vector = word2vec_model[word]
            embedding_matrix[index] = vector
        except:
            pass

    model = lstm(vocab_size, embedding_dim, embedding_matrix)
    y_train = train_df[LABELS]
    model, history = train_model(model, X_train, y_train)

    y_test = model.predict([X_test], batch_size=1024, verbose=1)
    sample_submission = pd.read_csv(f'data/sample_submission.csv.zip')
    sample_submission[LABELS] = y_test
    sample_submission.to_csv(f'submission_LSTM_word2vec.csv', index=False)


def lstm(vocab_size, embedding_dim, embedding_matrix):
    model = Sequential()
    model.add(
        Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=MAX_LENGTH, trainable=False))
    model.add(LSTM(60, return_sequences=True))
    model.add(GlobalMaxPool1D())
    model.add(Dropout(0.1))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(6, activation="sigmoid"))

    return model


def train_model(model, X_train, y_train):
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=2, validation_split=0.1, batch_size=64, verbose=1)

    return model, history


def clean_comment(comment):
    comment = comment.lower()
    comment = remove_special_chars(comment)
    comment = remove_stop_words(comment)

    return comment


def remove_special_chars(comment):
    comment = re.sub("(\\W)", " ", comment).strip()
    comment = re.sub('\S*\d\S*\s*', '', comment).strip()
    comment = re.sub(' +', ' ', comment)

    return comment


def remove_stop_words(comment):
    comment = " ".join([word for word in comment.split() if word not in stopwords.words('english')])

    return comment


if __name__ == '__main__':
    main()
