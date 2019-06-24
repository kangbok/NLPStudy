import pickle


class DataSetMaker:
    def __init__(self):
        pass

    def load_all_raw_data(self):
        print("raw data loading...")
        raw_data = []

        # 40만개만 이용하기
        for i in range(0, 500000, 100000):
            file_path = "../../../crawler/dataset/translation_%s.pkl" % i

            with open(file_path, "rb") as f:
                dataset_part = pickle.load(f)
                raw_data.extend(dataset_part)

        print("raw data is loaded!")

        return raw_data

    def save_dataset(self, raw_data, save_dir_path):
        print("dataset pickle files making...")

        kor_data, eng_data = self.split_kor_eng(raw_data, True)

        with open(save_dir_path + "/dataset_kor.pkl", "wb") as f:
            pickle.dump(kor_data, f)
        with open(save_dir_path + "/dataset_eng.pkl", "wb") as f:
            pickle.dump(eng_data, f)

        print("dataset pickle files is made!")

    def split_kor_eng(self, raw_data, is_format_change=True):
        print("kor-eng data splitting...")

        cnt = 0
        kor_list = []
        eng_list = []

        if is_format_change:
            for kor_data, eng_data in raw_data:
                kor_list.append(list(map(lambda x:"/".join(x), kor_data)))
                eng_list.append(list(map(lambda x:"/".join(x), eng_data)))

                if cnt % 20000 == 0:
                    print(cnt)

                cnt += 1
        else:
            for kor_data, eng_data in raw_data:
                kor_list.append(kor_data)
                eng_list.append(eng_data)

                if cnt % 20000 == 0:
                    print(cnt)

                cnt += 1

        "all kor-eng data is split!"

        return kor_list, eng_list


if __name__ == "__main__":
    ds = DataSetMaker()
    raw_data = ds.load_all_raw_data()
    ds.save_dataset(raw_data, "../dataset")
