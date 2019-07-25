import pickle

with open("../resource/ratings_train.txt", "r", encoding="utf-8") as f:
    line_list = f.readlines()

# 헤더 제외
line_list = line_list[1:]

answer_list = []

for s in line_list:
    answer = int(s.split("\t")[2].replace("\n", ""))
    answer_list.append(answer)

with open("../resource/answer_train.pkl", "wb") as f:
    pickle.dump(answer_list, f)
