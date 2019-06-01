import pickle

with open("../resource/sentence_matrixs_train.pkl", "rb") as f:
    corpus = pickle.load(f)

with open("../resource/answer_train.pkl", "rb") as f:
    answer_list = pickle.load(f)

if len(corpus) != len(answer_list):
    print("%s, %s, data is not matched!!" % (len(corpus), len(answer_list)))
    exit()

idx_list = []

for idx, matrix in enumerate(corpus):
    if len(matrix) == 0:
        idx_list.append(idx)

idx_list.sort(reverse=True)

for idx in idx_list:
    del corpus[idx]
    del answer_list[idx]


with open("../dataset/sentence_matrix_train.pkl", "wb") as f:
    pickle.dump(corpus, f)

with open("../dataset/answer_train.pkl", "wb") as f:
    pickle.dump(answer_list, f)
