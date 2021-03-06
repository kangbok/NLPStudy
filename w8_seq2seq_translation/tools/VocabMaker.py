#
# DataSetMaker.py 에서 만든 데이터셋에 대한 vocab 사전을 만드는 코드
#

import pickle


def create_vocab_dict(dataset_file_path, vocab_idx_path, idx_vocab_path, mode="eojeol"):
    with open(dataset_file_path, "rb") as f:
        dataset = pickle.load(f)

    vocab_idx_dict = {"<S>": 0, "<E>": 1, "<P>": 2}
    idx_vocab_dict = {0: "<S>", 1: "<E>", 2: "<P>"}
    idx = 3

    target_list = []
    if mode == "eojeol":
        for text in dataset:
            eojeol_list = text.split(" ")
            target_list.extend(eojeol_list)
    else:
        for word_list in dataset:
            target_list.extend(word_list)

    all_list = list(set(target_list))
    all_list.sort()

    for word in all_list:
        vocab_idx_dict[word] = idx
        idx_vocab_dict[idx] = word

        idx += 1

    with open(vocab_idx_path, "wb") as f:
        pickle.dump(vocab_idx_dict, f)

    with open(idx_vocab_path, "wb") as f:
        pickle.dump(idx_vocab_dict, f)


# create_vocab_dict("../dataset/dataset_word_kor.pkl", "../resource/vocab_idx_word_kor.pkl", "../resource/idx_vocab_word_kor.pkl", "word")
create_vocab_dict("../dataset/dataset_word_eng.pkl", "../resource/vocab_idx_word_eng.pkl", "../resource/idx_vocab_word_eng.pkl", "word")

