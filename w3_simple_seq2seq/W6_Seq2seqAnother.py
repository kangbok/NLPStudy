#
# tensorflow의 contrib 패키지 내의 seq2seq 패키지를 이용하여 구현한 seq2seq.
#

import os
import pickle

import numpy as np
import tensorflow as tf
import tensorflow.contrib as contrib

with open("../resource/vocab_idx_dict.pkl", "rb") as f:
    vocab_idx_dict = pickle.load(f)

with open("../resource/idx_vocab_dict.pkl", "rb") as f:
    idx_vocab_dict = pickle.load(f)

batch_size = 100
vocab_size = len(vocab_idx_dict.keys())
emb_size = 100
encoder_lstm_units = 256
decoder_lstm_units = 256
encoder_length = 1
decoder_length = 100

### place holders
encoder_x = tf.placeholder(tf.int32, [None, encoder_length], name="encoder_x")
decoder_x = tf.placeholder(tf.int32, [None, decoder_length], name="decoder_x")
real_decoder_length = tf.placeholder(tf.int32, [None, ], name="real_decoder_length")

##### encoder
encoder_emb_w = tf.get_variable('enc_embedding', initializer=tf.random_uniform([vocab_size, emb_size]), dtype=tf.float32)
enc_emb_inputs = tf.nn.embedding_lookup(encoder_emb_w, encoder_x, name='emb_inputs')

encoder_cell = tf.nn.rnn_cell.LSTMCell(encoder_lstm_units, name="encoder_cell")
encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell, enc_emb_inputs, dtype=tf.float32, scope="encoder_rnn")


##### decoder(seq2seq)
decoder_emb_w = tf.get_variable('dec_embedding', initializer=tf.random_uniform([vocab_size, emb_size]), dtype=tf.float32)
decoder_cell = tf.nn.rnn_cell.LSTMCell(decoder_lstm_units, name="decoder_cell")
output_layer = tf.layers.Dense(vocab_size, name='output_projection')

### training part
decoder_max_length = tf.reduce_max(real_decoder_length + 1, name="decoder_max_length")
dec_emb_inputs = tf.nn.embedding_lookup(decoder_emb_w, decoder_x, name='emb_inputs')
training_helper = contrib.seq2seq.TrainingHelper(inputs=dec_emb_inputs, sequence_length=real_decoder_length + 1, name="training_helper")
training_decoder = contrib.seq2seq.BasicDecoder(decoder_cell, training_helper, encoder_state, output_layer)

decoder_output_train, decoder_state_train, \
decoder_output_length_train = contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                             maximum_iterations=decoder_max_length,
                                                             impute_finished=True)

decoder_logits_train = tf.identity(decoder_output_train.rnn_output, name="decoder_logits_train")
targets = tf.slice(decoder_x, [0, 0], [-1, decoder_max_length], name="targets")
mask = tf.sequence_mask(real_decoder_length + 1, decoder_max_length, tf.float32, name="mask")
batch_loss = contrib.seq2seq.sequence_loss(logits=decoder_logits_train, targets=targets, weights=mask, name="loss")
optimizer = tf.train.AdamOptimizer(0.001).minimize(batch_loss)
valid_predictions = tf.identity(decoder_output_train.sample_id, name='valid_preds')

### inference part
batch_size_a = tf.shape(encoder_x)[0]
start_tokens = tf.zeros(batch_size_a, tf.int32, name="start_tokens")
inference_helper = contrib.seq2seq.GreedyEmbeddingHelper(embedding=decoder_emb_w,
                                                         start_tokens=start_tokens,
                                                         end_token=1)
inference_decoder = contrib.seq2seq.BasicDecoder(decoder_cell, inference_helper, encoder_state, output_layer)
decoder_output_inf, decoder_state_inf, \
decoder_output_length_inf = contrib.seq2seq.dynamic_decode(decoder=inference_decoder,
                                                           maximum_iterations=decoder_length,
                                                           impute_finished=True)
predictions = tf.identity(decoder_output_inf.sample_id, name='predictions')


################################################################################# 전처리 함수
# 문장을 word idx의 리스트로 바꿔줌
def create_word_idx_list(stc_list, batch):
    out_encoder_list = []
    out_decoder_list = []

    for word_list in stc_list:
        one_encoder_list = []
        one_decoder_list = []

        for idx, word in enumerate(word_list):
            if idx == 0:
                one_encoder_list.append(vocab_idx_dict[word])
            else:
                one_decoder_list.append(vocab_idx_dict[word])

        out_encoder_list.append(one_encoder_list)
        out_decoder_list.append(one_decoder_list)

    out_encoder_list = np.asarray(out_encoder_list, dtype="int32")
    out_encoder_list = out_encoder_list.reshape([batch,-1])

    return out_encoder_list, out_decoder_list


def rectify_decoder_input(word_idx_list, batch):
    out_decoder_i = []
    out_decoder_size = []

    for one_list in word_idx_list:
        decoder_i = np.append(0, np.asarray(one_list)) # 0의 의미 : <S> 토큰
        decoder_size = len(one_list)

        # 100으로 사이즈 맞춰줄 때 쓰던 부분
        for _ in range(100 - len(decoder_i)):
            decoder_i = np.append(decoder_i, 1) # 1의 의미 : <E> 토큰

        out_decoder_i.append(decoder_i)
        out_decoder_size.append(decoder_size)

    out_decoder_i = np.asarray(out_decoder_i, dtype=np.int).reshape((batch, -1))
    out_decoder_size = np.asarray(out_decoder_size, dtype=np.int).reshape((batch, ))

    return out_decoder_i, out_decoder_size

################################################################################# 모델 저장
saver = tf.train.Saver()
SAVER_DIR = "../model"
checkpoint_path = os.path.join(SAVER_DIR, "seq2seq_model")
ckpt = tf.train.get_checkpoint_state(SAVER_DIR)


################################################################################# 데이터 불러오기
with open("../resource/corpus_train_wo_frequent_words.pkl", "rb") as f:
    corpus = pickle.load(f)

def print_tmp_result(x_input, predicted_result):
    input_word = idx_vocab_dict[np.max(x_input[0])]

    predicted_result = predicted_result.reshape(100, -1)

    output_list = [idx_vocab_dict[int(i)] for i in predicted_result[0]]
    print("%s -> %s" % (input_word, output_list))

    with open("예시.txt", "a") as f:
        f.write("%s -> %s\n" % (input_word, output_list))


################################################################################# 실행
# 훈련용, 테스트용 데이터 분리
cutting_point = 150000
train_x = corpus[:cutting_point]
test_x = corpus[cutting_point:]


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    BATCH_SIZE = 100
    cnt = 0

    for epoch in range(5000):
        for idx in range(0, len(train_x), BATCH_SIZE):
            input_words = train_x[idx:idx + BATCH_SIZE]

            encoder_x_, decoder_idx_list = create_word_idx_list(input_words, BATCH_SIZE)
            decoder_x_, decoder_size_ = rectify_decoder_input(decoder_idx_list, BATCH_SIZE)

            # 매 100개 batch마다 step print, 모델 저장
            if cnt % 100 == 0:
                test_result = sess.run(predictions, feed_dict={encoder_x: encoder_x_, real_decoder_length: decoder_size_})
                print("step %s" % cnt)
                print(input_words[0])
                print_tmp_result(encoder_x_, test_result)
                saver.save(sess, checkpoint_path)

            sess.run(optimizer, feed_dict={encoder_x: encoder_x_, decoder_x: decoder_x_, real_decoder_length: decoder_size_})

            cnt += 1
