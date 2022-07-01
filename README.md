# Toxic Comment Classification - IASD
Our participation to the Toxic Comment Classification Kaggle Challenge as final project for the IASD NLP course. \

http://github.com – automatic! [GitHub](http://github.com) \

\href{https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge}{Kaggle competition} proposed by Jigsaw, a Google's subsidaries belonging to Alphabet. In this competition, the challenge was to build a multi-headed model that’s capable of detecting different types of toxicity like threats, obscenity, insults, and identity-based hate better than Perspective’s models in  2018. Perspective is an API developed by the Conversation AI team, a research initiative founded by Jigsaw, using machine learning to reduce toxicity online. \
For us, the challenge was to use and implement the different classification methods we saw during the NLP course. We decided to use the pretrained embedding GloVe 6B with 4 different vector dimmensions: 50, 100, 200 and 300. We will try all of them and compare the results. For the model, we decided to try 5 different architectures: CNN, LSTM, bidirectional LSTM, GRU and bidirectional GRU to compare the results and find the best architecture for this classification task. 
