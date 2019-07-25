#
# resource/HandMadeOneDocument.json 파일을 불러와 데이터셋 형태로 변환하는 것.
#

import json
import pickle

from konlpy.tag import Komoran


class DatasetMakerOld:
    def __init__(self, file_path):
        self.source_list = self.create_source_list(file_path)
        self.tokenizer = Komoran()

    def create_source_list(self, file_path):
        source_json_dict = self.load_json(file_path)

        kor_list = []
        eng_list = []

        # title 처리
        title_dict = source_json_dict["title"]
        kor_list.append(title_dict["kor"])
        eng_list.append(title_dict["eng"])

        # update_dates 처리
        ud_dict = source_json_dict["update_dates"]

        for kor_text in ud_dict["kor"]:
            kor_list.append(kor_text)
        for eng_text in ud_dict["eng"]:
            eng_list.append(eng_text)

        # articles 처리
        for article in source_json_dict["articles"]:
            for sub_article in article:
                contents_dict = sub_article["contents"]

                kor_list.append(contents_dict["kor"])
                eng_list.append(contents_dict["eng"])

        return list(zip(kor_list, eng_list))

    def load_json(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            out_json = json.load(f)

        return out_json

    def save_dataset(self, save_dir_path, mode="word"):
        if mode == "word":
            kor_dataset, eng_dataset = self.make_word_set()
        elif mode == "char":
            kor_dataset, eng_dataset = self.make_char_set()
        else:
            kor_dataset, eng_dataset = self.make_eojeol_set()

        with open("%s/%s_kor.pkl" % (save_dir_path, mode), "wb") as f:
            pickle.dump(kor_dataset, f)

        with open("%s/%s_eng.pkl" % (save_dir_path, mode), "wb") as f:
            pickle.dump(eng_dataset, f)


    def make_word_set(self):
        return [], []

    def make_char_set(self):
        return [], []

    def make_eojeol_set(self):
        kor_dataset = []
        eng_dataset = []

        for kor, eng in self.source_list:
            kor_dataset.append(kor)
            eng_dataset.append(eng)

        return kor_dataset, eng_dataset


if __name__ == "__main__":
    file_path = "../resource/HandMadeOneDocument.json"

    dm = DatasetMakerOld(file_path)
    dm.save_dataset("../dataset", "eojeol")

