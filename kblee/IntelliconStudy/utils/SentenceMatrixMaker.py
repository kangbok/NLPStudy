import pickle

import numpy as np

## load data
from gensim.models import Word2Vec

with open("../resource/corpus_train.pkl", "rb") as f:
    corpus = pickle.load(f)

with open("../resource/answer_train.pkl", "rb") as f:
    answers = pickle.load(f)

## load word vector
w2v = Word2Vec.load("../model/wordvector_w2v.model")


matrix_list = []

for word_list in corpus:
    vector_list = []

    for word in word_list:
        try:
            v = w2v.wv.get_vector(word)
            vector_list.append(v)
        except:
            continue

    if vector_list:
        sentence_matrix = np.concatenate(vector_list).reshape((-1, 300))
        matrix_list.append(sentence_matrix)
    else:
        matrix_list.append([])

with open("../resource/sentence_matrixs_train.pkl", "wb") as f:
    pickle.dump(matrix_list, f)
