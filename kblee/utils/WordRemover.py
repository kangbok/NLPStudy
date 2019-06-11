import pickle


def get_removing_condition(token):
    return token == "영화/Noun" or "/Punctuation" in token


with open("../resource/corpus_train.pkl", "rb") as f:
    corpus = pickle.load(f)


new_corpus = []

for token_list in corpus:
    new_token_list = []

    for token in token_list:
        condition = get_removing_condition(token)

        if not condition:
            new_token_list.append(token)

    if new_token_list:
        new_corpus.append(new_token_list)


with open("../resource/corpus_train_wo_frequent_words.pkl", "wb") as f:
    pickle.dump(new_corpus, f)
