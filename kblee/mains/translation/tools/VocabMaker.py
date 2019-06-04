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


create_vocab_dict("../dataset/eojeol_kor.pkl", "../resource/vocab_idx_eojeol_kor.pkl", "../resource/idx_vocab_eojeol_kor.pkl", "eojeol")
create_vocab_dict("../dataset/eojeol_eng.pkl", "../resource/vocab_idx_eojeol_eng.pkl", "../resource/idx_vocab_eojeol_eng.pkl", "eojeol")

