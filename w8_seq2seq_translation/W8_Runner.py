#
# 파일로 저장한 모델을 실행시켜서 번역을 진행하는 코드.
# 소량의 데이터만 이용했으므로, W11_Runner를 보는 것을 추천.
#

import pickle

import numpy as np
import tensorflow as tf

with open("resource/vocab_idx_eojeol_kor.pkl", "rb") as f:
    vocab_idx_dict = pickle.load(f)

with open("resource/idx_vocab_eojeol_eng.pkl", "rb") as f:
    idx_vocab_dict = pickle.load(f)


with tf.Session() as sess:
    loader = tf.train.import_meta_graph("model/seq2seq_model.meta")
    loader.restore(sess, tf.train.latest_checkpoint("model/"))

    graph = tf.get_default_graph()
    predictions = graph.get_tensor_by_name("predictions:0")
    encoder_x = graph.get_tensor_by_name("encoder_x:0")

    sentence = "해양환경의 보호 및 보전"
    word_list = sentence.split(" ")
    input_x = list(map(lambda x:vocab_idx_dict[x], word_list))
    INPUT_LENGTH = 35

    for _ in range(INPUT_LENGTH - len(input_x)):
        input_x.append(2)

    input_x = np.asarray(input_x).reshape(1, -1)

    result = sess.run(predictions, feed_dict={encoder_x: input_x})

    for row in result:
        print(list(map(lambda x:idx_vocab_dict[x], row)))
