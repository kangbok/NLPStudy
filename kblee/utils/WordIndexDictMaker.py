import pickle

##################################################################### data loading
with open("../resource/corpus_test.pkl", "rb") as f:
    corpus_test = pickle.load(f)

print("test data loading!")

with open("../resource/corpus_train_wo_frequent_words.pkl", "rb") as f:
    corpus_training = pickle.load(f)

print("training data loading!")

##################################################################### test, traing 데이터 하나로 합침
all_sentence_list = []
# all_sentence_list.extend(corpus_test)
all_sentence_list.extend(corpus_training)
all_sentence_list = all_sentence_list[:10]

##################################################################### 단어 사전 만듬
all_word_list = []
vocab_idx_dict = {"<S>": 0, "<E>": 1, "<P>": 2}
idx_vocab_dict = {0: "<S>", 1: "<E>", 2: "<P>"}
idx = 3

for word_list in all_sentence_list:
    all_word_list.extend(word_list)

all_word_list = list(set(all_word_list))
all_word_list.sort()

for word in all_word_list:
    vocab_idx_dict[word] = idx
    idx_vocab_dict[idx] = word

    idx += 1


with open("../resource/vocab_idx_dict10.pkl", "wb") as f:
    pickle.dump(vocab_idx_dict, f)

with open("../resource/idx_vocab_dict10.pkl", "wb") as f:
    pickle.dump(idx_vocab_dict, f)
