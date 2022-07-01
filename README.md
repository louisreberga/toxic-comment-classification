# Toxic Comment Classification

This project is our participation to the [Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge), a Kaggle competition proposed by Jigsaw, a Google's subsidaries belonging to Alphabet. In this competition, the challenge was to build a multi-headed model that is capable of detecting different types of toxicity like threats, obscenity, insults, and identity-based hate better than Perspective’s models in 2018. Perspective is an API developed by the Conversation AI team, a research initiative founded by Jigsaw, using machine learning to reduce toxicity online. \
For us, the challenge was to use and implement the different classification methods we saw during the [IASD](https://www.lamsade.dauphine.fr/wp/iasd/)'s NLP course master. We decided to use as embedding the pretrained GloVe 6B with 4 different vector dimmensions: 50, 100, 200 and 300. For the model, we decided to try 5 different architectures: CNN, LSTM, bidirectional LSTM, GRU and bidirectional GRU to compare the results and find the best architecture for this classification task. 

Here is our results, the models are evaluated with a mean column-wise ROC AUC:

| Embedding | CNN | LSTM | Bidirectional LSTM | GRU | Bidirectional GRU |
| --- | --- | --- | --- |--- |--- |
| GloVe 6B  50D | 0.96677 | 0.97146 | 0.97389 | 0.9712 | 0.97373 |
| GloVe 6B 100D | 0.97205 | 0.97415 | 0.9778 | 0.97322 | 0.97742 |
| GloVe 6B 200D | 0.97147 | 0.97599 | 0.97805 | 0.97702 | 0.97828 |
| GloVe 6B 300D | 0.97249 | 0.97612 | 0.97898 | 0.97669 | 0.97855 |
