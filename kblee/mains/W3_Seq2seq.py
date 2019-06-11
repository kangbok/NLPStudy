import os
import pickle

import numpy as np
import tensorflow as tf

with open("../resource/vocab_idx_dict.pkl", "rb") as f:
    vocab_idx_dict = pickle.load(f)

with open("../resource/idx_vocab_dict.pkl", "rb") as f:
    idx_vocab_dict = pickle.load(f)

batch_size = 1
vocab_size = len(vocab_idx_dict.keys())
encoder_lstm_units = 256
decoder_lstm_units = 256

encoder_x = tf.placeholder(tf.float32, [1, None, vocab_size])
# decoder_x = tf.placeholder(tf.float32, [1, None, vocab_size])
decoder_y = tf.placeholder(tf.int32, [1, None, vocab_size])

##### encoder
encoder_cell = tf.nn.rnn_cell.LSTMCell(encoder_lstm_units)
encoder_dropout_cell = tf.nn.rnn_cell.DropoutWrapper(encoder_cell, output_keep_prob=0.75)
encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_dropout_cell, encoder_x, dtype=tf.float32, scope="encoder_rnn")

##### decoder functions
W = tf.Variable(tf.truncated_normal(shape=[decoder_lstm_units, vocab_size], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[vocab_size]))

def loop_fn(time, cell_output, cell_state, loop_state):
    if cell_output is None: # time = 0 이라는 뜻
        return loop_fn_initial()
    else:
        return loop_fn_transition(time, cell_output, cell_state, loop_state)

def loop_fn_initial():
    finish = 0 >= tf.constant(100, dtype=tf.int32)
    inp = tf.reshape(tf.one_hot([vocab_idx_dict["<S>"]], vocab_size), (1, vocab_size))
    cell_state = encoder_state
    emit_output = None
    loop_state = None  # 뭔가 추가 정보를 넘길 때 사용하는 것인듯.

    return finish, inp, cell_state, emit_output, loop_state

def loop_fn_transition(time, prev_output, prev_state, prev_loop_state):
    output_logits = tf.add(tf.matmul(prev_output, W), b)
    prediction = tf.argmax(output_logits, axis=1)

    def finish_true_fn():
        return time >= tf.constant(0, dtype=tf.int32)
    def finish_false_fn():
        return time >= tf.constant(100, dtype=tf.int32)

    finish_condition = tf.equal(prediction, tf.reshape(tf.constant(1, dtype=tf.int64), (1,)))
    finish = tf.cond(tf.reshape(finish_condition, ()), finish_true_fn, finish_false_fn)

    def input_true_fn():
        return tf.zeros((1, vocab_size))
    def input_false_fn():
        return tf.reshape(tf.one_hot([prediction[0]], vocab_size), (1, vocab_size))

    fin = time >= tf.constant(100, dtype=tf.int32)
    inp = tf.cond(finish, input_true_fn, input_false_fn)
    state = prev_state
    emit_output = prev_output
    loop_state = prev_loop_state

    return fin, inp, state, emit_output, loop_state


##### decoder
decoder_cell = tf.nn.rnn_cell.LSTMCell(decoder_lstm_units)
decoder_dropout_cell = tf.nn.rnn_cell.DropoutWrapper(decoder_cell, output_keep_prob=0.75)
decoder_output_ta, decoder_state, decoder_tmp = tf.nn.raw_rnn(decoder_dropout_cell, loop_fn, scope="decoder_rnn")
decoder_outputs = decoder_output_ta.stack()

##### 학습 영역
# final_length = tf.minimum(tf.shape(decoder_y)[1], tf.shape(decoder_outputs)[0])
filtered_decoder_outputs = tf.gather(decoder_outputs, tf.range(0, tf.shape(decoder_y)[1]))
decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(filtered_decoder_outputs))
decoder_outputs_flat = tf.reshape(filtered_decoder_outputs, (-1, decoder_dim))
decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))

# filtered_decoder_y = tf.gather(decoder_y, tf.range(0, final_length), axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=decoder_y, logits=decoder_logits)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

##### 예측 영역
decoder_prediction = tf.argmax(decoder_logits, 2)


################################################################################# 전처리 함수
# 단어를 one-hot vector로 바꿔줌
def create_one_hot_matrix(stc_list):
    out_list = []

    for word_list in stc_list:
        for word in word_list:
            one_hot_vector = np.zeros((vocab_size, ), dtype=int)
            one_hot_vector[vocab_idx_dict[word]] = 1
            one_hot_vector.reshape((1, vocab_size))
            out_list.append(one_hot_vector)

    return np.array(out_list)

def padding_one_hot():
    one_hot_vector = np.zeros((vocab_size,), dtype=int)
    one_hot_vector[2] = 1

    return one_hot_vector


def create_encoder_decoder_io(one_hot_matrix):
    encoder_i = one_hot_matrix[0]

    start_char_vector = np.zeros((vocab_size,), dtype=int)
    start_char_vector[0] = 1
    end_char_vector = np.zeros((vocab_size,), dtype=int)
    end_char_vector[1] = 1

    decoder_i = np.append(start_char_vector, one_hot_matrix[1:])
    # decoder_o = np.append(one_hot_matrix[1:], end_char_vector)
    decoder_o = one_hot_matrix[1:]

    # 100으로 사이즈 맞춰줄 때 쓰던 부분
    # for _ in range(100 - int((decoder_o.shape[0] / vocab_size))):
    #     decoder_o = np.append(decoder_o, padding_one_hot())

    encoder_i = encoder_i.reshape((1, 1, -1))
    decoder_i = decoder_i.reshape((1, -1, vocab_size))
    decoder_o = decoder_o.reshape((1, -1, vocab_size))

    return encoder_i, decoder_i, decoder_o

################################################################################# 모델 저장
saver = tf.train.Saver()
SAVER_DIR = "../model"
checkpoint_path = os.path.join(SAVER_DIR, "seq2seq_model")
ckpt = tf.train.get_checkpoint_state(SAVER_DIR)


################################################################################# 데이터 불러오기
with open("../resource/corpus_train_wo_frequent_words.pkl", "rb") as f:
    corpus = pickle.load(f)

def print_tmp_result(x_input, predicted_result):
    input_word = idx_vocab_dict[np.argmax(x_input)]
    output_list = [idx_vocab_dict[int(i)] for i in predicted_result]
    print("%s -> %s" % (input_word, output_list))

    with open("데이터10-단어수6만.txt", "a") as f:
        f.write("%s -> %s\n" % (input_word, output_list))


################################################################################# 실행
# 훈련용, 테스트용 데이터 분리
cutting_point = 10
train_x = corpus[:cutting_point]
test_x = corpus[cutting_point:]


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    BATCH_SIZE = 1
    cnt = 0

    for epoch in range(5000):
        for idx in range(0, len(train_x), BATCH_SIZE):
            input_words = train_x[idx:idx + BATCH_SIZE]

            one_hot_matrix = create_one_hot_matrix(input_words)

            if len(one_hot_matrix) <= 1:
                continue

            encoder_x_, decoder_x_, decoder_y_ = create_encoder_decoder_io(one_hot_matrix)
            # ccc = sess.run(decoder_outputs_flat, feed_dict={encoder_x: encoder_x_, decoder_y: decoder_y_})
            #
            # print("a")

            # 매 10개마다 step print, 모델 저장
            # if cnt % 10 == 0:
            #     test_result = sess.run(decoder_prediction, feed_dict={encoder_x: encoder_x_, decoder_y: decoder_y_})
            #     print("step %s" % cnt)
            #     print(input_words)
            #     print_tmp_result(encoder_x_, test_result)

            test_result = sess.run(decoder_prediction, feed_dict={encoder_x: encoder_x_, decoder_y: decoder_y_})
            print("step %s" % cnt)
            print(input_words)

            with open("데이터10-단어수6만.txt", "a") as f:
                f.write("step %s\n" % cnt)
                f.write(" ".join(input_words[0]) + "\n")

            print_tmp_result(encoder_x_, test_result)

                # saver.save(sess, checkpoint_path)

            sess.run(optimizer, feed_dict={encoder_x: encoder_x_, decoder_y: decoder_y_})

            cnt += 1
