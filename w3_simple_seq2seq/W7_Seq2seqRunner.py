import pickle

import tensorflow as tf

with open("../resource/vocab_idx_dict.pkl", "rb") as f:
    vocab_idx_dict = pickle.load(f)

with open("../resource/idx_vocab_dict.pkl", "rb") as f:
    idx_vocab_dict = pickle.load(f)


with tf.Session() as sess:
    loader = tf.train.import_meta_graph("../model/seq2seq_model.meta")
    loader.restore(sess, tf.train.latest_checkpoint("../model/"))

    graph = tf.get_default_graph()
    predictions = graph.get_tensor_by_name("predictions:0")
    encoder_x = graph.get_tensor_by_name("encoder_x:0")
    real_decoder_length = graph.get_tensor_by_name("real_decoder_length:0")

    word = "송강호/Noun"
    input_x = [[vocab_idx_dict[word]]]
    decoder_size = [10]

    result = sess.run(predictions, feed_dict={encoder_x: input_x, real_decoder_length: decoder_size})

    for row in result:
        for word in row:
            if word == 1:
                break

            print(idx_vocab_dict[word])
