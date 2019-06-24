import pickle

### Korean dataset loading
with open("../dataset/dataset_word_kor.pkl", "rb") as f:
    kor_datset = pickle.load(f)


### English dataset loading
with open("../dataset/dataset_word_eng.pkl", "rb") as f:
    eng_datset = pickle.load(f)


### Korean vocab dictionary loading
with open("../resource/vocab_idx_word_kor.pkl", "rb") as f:
    kor_vocab = pickle.load(f)


### English vocab dictionary loading
with open("../resource/vocab_idx_word_eng.pkl", "rb") as f:
    eng_vocab = pickle.load(f)


print("Korean dataset size : %s" % len(kor_datset))
print("English dataset size : %s" % len(eng_datset))

print("Korean vocab size : %s" % len(kor_vocab))
print("English vocab size : %s" % len(eng_vocab))

print("Korean max word count : %s" % max(map(lambda x:len(x), kor_datset)))
print("English max word count : %s" % max(map(lambda x:len(x), eng_datset)))
