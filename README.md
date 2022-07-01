# Toxic Comment Classification - IASD

This project is our participation to the [Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge), a Kaggle competition proposed by Jigsaw, a Google's subsidaries belonging to Alphabet. In this competition, the challenge was to build a multi-headed model that is capable of detecting different types of toxicity like threats, obscenity, insults, and identity-based hate better than Perspective’s models in 2018. Perspective is an API developed by the Conversation AI team, a research initiative founded by Jigsaw, using machine learning to reduce toxicity online. \
For us, the challenge was to use and implement the different classification methods we saw during the [IASD](https://www.lamsade.dauphine.fr/wp/iasd/) NLP course. We decided to use as embedding the pretrained GloVe 6B with 4 different vector dimmensions: 50, 100, 200 and 300. For the model, we decided to try 5 different architectures: CNN, LSTM, bidirectional LSTM, GRU and bidirectional GRU to compare the results and find the best architecture for this classification task. 

Here is our results:
