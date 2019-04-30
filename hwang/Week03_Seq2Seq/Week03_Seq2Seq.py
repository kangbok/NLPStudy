import pickle
import time
import sys
import csv

# Third-party packages
import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec

# local module in the same directory
from BatchGenerator import BatchGenerator


train_morph_file_path = "../data/corpus_train.pkl"
test_morph_file_path = "../data/corpus_test.pkl"

train_ans_file_path = "../data/ratings_train.txt"
test_ans_file_path = "../data/ratings_test.txt"

def get_data_from_pickle(pickle_file_name):
    time_start = time.time()
    sys.stdout.write("Loading Data from \"%s\" ... " % pickle_file_name)
    with open(pickle_file_name, 'rb') as f:
        docs = pickle.load(f, encoding='utf8')
    time_end = time.time()
    sys.stdout.write("Done (%d seconds)\n" % int(time_end - time_start))

    return docs


def get_data_from_tsv(tsv_file_name):
    time_start = time.time()
    sys.stdout.write("Loading Data from \"%s\" ... " % tsv_file_name)
    with open(tsv_file_name, 'r', encoding='utf8') as f:
        docs = csv.reader(f, delimiter='\t')
        docs = list(docs)
        docs = docs[1:]
        docs = [label for _, _, label in docs]

    time_end = time.time()
    sys.stdout.write("Done (%d seconds)\n" % int(time_end - time_start))

    return docs


if __name__ == "__main__":
    # load train data
    train_morph_data = get_data_from_pickle(train_morph_file_path)
    print(len(train_morph_data))
    print("See a Train Data Sample")
    print(train_morph_data[0])
    print()

    train_ans_data = get_data_from_tsv(train_ans_file_path)
    train_ans_data = [int(ans) for ans in train_ans_data]
    print(len(train_ans_data))
    print("See a Train Answer Sample")
    print(type(train_ans_data[0]))
    print(train_ans_data[0])
    print()

    all_unique_morphs = set()
    for sent in train_morph_data:
        all_unique_morphs.update(sent)

    all_unique_morphs = list(all_unique_morphs)
    print(len(all_unique_morphs))



