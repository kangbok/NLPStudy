import pickle
import re

from konlpy.tag import Komoran


class Preprocessor:
    def __init__(self):
        self.tokenizer = Komoran()

    def get_raw_data(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = f.readlines()

        return raw_data

    def is_unused_line(self, text):
        """
        데이터셋에 넣지 않을 라인인지 확인
        :param text: 텍스트
        :return: 넣지 않을 지 여부
        """
        out_bool = False

        if text[0:5] == "[본조신설" or text[0:5] == "[전문개정" or text[1:4] == " 삭제":
            out_bool = True

        return out_bool

    def remove_unsued_part(self, text, is_kor=True):
        """
        불필요한 내용 삭제
        :param text: 텍스트
        :param is_kor: 한글 처리인지 여부 (True : 한글 처리, False : 영어 처리)
        :return: 불필요한 부분이 제거된 텍스트
        """
        if is_kor:
            out_text = re.sub(r"[<]개정.*[>]", "", text)

            try:
                if 9312 <= ord(out_text[0]) <= 9332:
                    out_text = out_text[1:]
            except IndexError as e:
                return text
        else:
            out_text = re.sub(r"[<]Amended.*[>]", "", text)
            out_text = re.sub(r"^[(][\d]+[)]", "", out_text)
            out_text = out_text.replace("\n", " ")

        return out_text

    def create_dataset(self, raw_data, save_file_path):
        print("만들기 시작!")

        output_list = []
        cnt = 0

        for line in raw_data:
            if self.is_unused_line(line):
                continue

            kor_eng = line.split("\t")

            if len(kor_eng) == 1:
                continue

            kor = kor_eng[0]
            eng = kor_eng[1]

            new_kor = self.remove_unsued_part(kor, True)
            new_eng = self.remove_unsued_part(eng, False)

            try:
                tokenized_kor_list = self.tokenize(new_kor)
                tokenized_eng_list = self.tokenize(new_eng)
            except Exception as e:
                continue

            output_list.append((tokenized_kor_list, tokenized_eng_list))

            if cnt % 5000 == 0:
                print(cnt)
                print(line.replace("\n", ""))

            cnt += 1

        print("만들기 종료!")

        with open(save_file_path, "wb") as f:
            pickle.dump(output_list, f)

    def tokenize(self, text):
        word_pos_list = self.tokenizer.pos(text)

        out_list = []

        for word, pos in word_pos_list:
            out_list.append((word, pos))

        return out_list

    def test_dataset_load(self, file_path):
        with open(file_path, "rb") as f:
            dataset = pickle.load(f)

        print(len(dataset))
        print(dataset[0])

if __name__ == "__main__":
    pp = Preprocessor()

    # 데이터셋 만들기
    datasource_file = "datasource/raw_text.txt"
    raw_data = pp.get_raw_data(datasource_file)

    for i in range(0, len(raw_data), 100000):
        if i < 200000:
            continue

        print("================= %s part start!" % i)

        save_file_path = "dataset/translation_%s.pkl" % i
        pp.create_dataset(raw_data[i:i + 100000], save_file_path)

    #데이터셋 잘 만들어졌는지 확인
    # dataset_file_path = "dataset/translation.pkl"
    # pp.test_dataset_load(dataset_file_path)
