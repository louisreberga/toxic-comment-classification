from constants import LABELS, MAX_LENGTH
import pandas as pd
from re import sub
from os import path, mkdir
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from pickle import load, dump, HIGHEST_PROTOCOL


def read_preprocessed_comments():
    directory = 'preprocessed_data'
    X_train = pd.read_csv(f'{directory}/X_train.csv')
    y_train = pd.read_csv(f'{directory}/y_train.csv')
    X_test = pd.read_csv(f'{directory}/X_test.csv')
    y_test = pd.read_csv(f'{directory}/y_test.csv')
    with open(f'{directory}/tokenizer.pickle', 'rb') as handle:
        tokenizer = load(handle)

    return X_train, y_train, X_test, y_test, tokenizer


def preprocess_comments():
    print('Retrieving data...')
    train_df = pd.read_csv('data/train.csv.zip')
    y_train = train_df[LABELS]

    test_df = pd.read_csv('data/test.csv.zip')
    test_labels = pd.read_csv('data/test_labels.csv.zip')
    test_labels = test_labels[test_labels["toxic"] != -1]
    test_df = test_df.merge(test_labels, on='id')
    y_test = test_df[LABELS]
    print('Data retrieved!\n')

    print('Cleaning data...')
    train_df['comment_text'] = train_df['comment_text'].apply(clean_comment)
    list_comment_train = list(train_df['comment_text'])

    test_df['comment_text'] = test_df['comment_text'].apply(clean_comment)
    list_comment_test = list(test_df['comment_text'])
    print('Data cleaned!\n')

    print('Tokenizing data...')
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(list_comment_train)
    encoded_comment_train = tokenizer.texts_to_sequences(list_comment_train)
    encoded_comment_test = tokenizer.texts_to_sequences(list_comment_test)
    vocab_size = len(tokenizer.word_index) + 1
    print('Data tokenized!\n')

    print('Padding data...')
    X_train = pad_sequences(encoded_comment_train, maxlen=MAX_LENGTH, padding='post')
    X_test = pad_sequences(encoded_comment_test, maxlen=MAX_LENGTH, padding='post')
    print('Data padded!\n')

    print('Saving data...')
    # print('vocab_size', vocab_size)

    directory = 'preprocessed_data'
    if not path.exists(directory):
        mkdir(directory)

    pd.DataFrame(X_train).to_csv(f'{directory}/X_train.csv', index=False)
    pd.DataFrame(y_train).to_csv(f'{directory}/y_train.csv', index=False)
    pd.DataFrame(X_test).to_csv(f'{directory}/X_test.csv', index=False)
    pd.DataFrame(y_test).to_csv(f'{directory}/y_test.csv', index=False)
    with open(f'{directory}/tokenizer.pickle', 'wb') as handle:
        dump(tokenizer, handle, protocol=HIGHEST_PROTOCOL)
    print('Data saved!')

    return X_train, y_train, X_test, y_test, tokenizer


def clean_comment(comment):
    comment = comment.lower()
    comment = remove_special_chars(comment)
    comment = remove_stop_words(comment)

    return comment


def remove_special_chars(comment):
    comment = sub("(\\W)", " ", comment).strip()
    comment = sub('\S*\d\S*\s*', '', comment).strip()
    comment = sub(' +', ' ', comment)

    return comment


def remove_stop_words(comment):
    return " ".join([word for word in comment.split() if word not in stopwords.words('english')])
