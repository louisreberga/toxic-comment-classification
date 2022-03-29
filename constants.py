LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

EMBEDDINGS = {
    "GloVe 6B 50D": "glove.6B.50d.txt",
    "GloVe 6B 100D": "glove.6B.100d.txt",
    "GloVe 6B 200D": "glove.6B.200d.txt",
    "GloVe 6B 300D": "glove.6B.300d.txt",
}

# PRETRAINED EMBEDDING PARAMETERS
EMBEDDING = 'GloVe 6B 100D'
EMB_DIM = 100
MAX_LENGTH = 400
VOCAB_SIZE = 169717

# MODEL HYPERPARAMETERS
MODEL = 'LSTM'
EPOCHS = 5
BATCH_SIZE = 64


