import pickle

import numpy as np
import tensorflow as tf
from konlpy.tag import Komoran

with open("resource/vocab_idx_word_kor.pkl", "rb") as f:
    vocab_idx_dict = pickle.load(f)

with open("resource/idx_vocab_word_eng.pkl", "rb") as f:
    idx_vocab_dict = pickle.load(f)

komoran = Komoran()

with tf.Session() as sess:
    loader = tf.train.import_meta_graph("model/seq2seq_attention_translation.meta")
    loader.restore(sess, tf.train.latest_checkpoint("model/"))

    graph = tf.get_default_graph()
    predictions = graph.get_tensor_by_name("predictions:0")
    encoder_x = graph.get_tensor_by_name("encoder_x:0")
    encoder_length = graph.get_tensor_by_name("real_encoder_length:0")

    sentence = "배타적 경제수역법"
    word_pos_list = komoran.pos(sentence)
    word_pos_list = ["/".join(t) for t in word_pos_list]
    input_x = list(map(lambda x:vocab_idx_dict[x], word_pos_list))
    INPUT_LENGTH = 300
    BATCH_SIZE = 30

    for _ in range(INPUT_LENGTH - len(input_x)):
        input_x.append(2)

    for _ in range(BATCH_SIZE - 1):
        for _ in range(INPUT_LENGTH):
            input_x.append(0)

    input_x = np.asarray(input_x).reshape(BATCH_SIZE, -1)
    input_length = [len(input_x)] * BATCH_SIZE

    result = sess.run(predictions, feed_dict={encoder_x: input_x, encoder_length: input_length})

    for row in result:
        print(list(map(lambda x:idx_vocab_dict[x], row)))
        break
