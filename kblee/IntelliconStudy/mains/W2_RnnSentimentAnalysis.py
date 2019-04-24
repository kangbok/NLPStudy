import pickle

import numpy as np
import tensorflow as tf


batch_size = 100
max_word_count = 100
embedding_size = 300
lstm_units = 128
class_count = 2

x = tf.placeholder(tf.float32, [None, max_word_count, embedding_size])
y = tf.placeholder(tf.float32, [None, class_count])

##### single-directional RNN
# cell = tf.nn.rnn_cell.LSTMCell(lstm_units)
# dropout_cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=0.75, output_keep_prob=0.75, state_keep_prob=1.0)
# output_rnn, state_rnn = tf.nn.dynamic_rnn(dropout_cell, x, dtype=tf.float32)
# output_rnn_t = tf.transpose(output_rnn, perm=[1, 0, 2]) # 가장 마지막 t에서의 output값만 얻기 위해 transpose 해줌
# last_outputs = tf.gather(output_rnn_t, int(output_rnn_t.get_shape()[0]) - 1) # 가장 마지막 t에서의 output값만 빼줌
# # output_rnn3 = output_rnn2[-1] ==> 위와 같은 내용의 코드.

##### stacked single-directional RNN
# cells = [tf.nn.rnn_cell.LSTMCell(lstm_units) for _ in range(2)]
# s_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
# output_rnn, state_rnn = tf.nn.dynamic_rnn(s_cell, x, dtype=tf.float32)
# output_rnn_t = tf.transpose(output_rnn, perm=[1, 0, 2]) # 가장 마지막 t에서의 output값만 얻기 위해 transpose 해줌
# last_outputs = tf.gather(output_rnn_t, int(output_rnn_t.get_shape()[0]) - 1) # 가장 마지막 t에서의 output값만 빼줌

##### single-directional RNN용 분류기용 가중치 매트릭스
# w = tf.Variable(tf.truncated_normal(shape=[lstm_units, class_count], stddev=0.1))
# b = tf.Variable(tf.constant(0.1, shape=[class_count]))


##### bi-directional RNN
# cell_forward = tf.nn.rnn_cell.LSTMCell(lstm_units)
# cell_backward = tf.nn.rnn_cell.LSTMCell(lstm_units)
# output_rnn, state_rnn = tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, x, dtype=tf.float32)
# output_concat = tf.concat(output_rnn, axis=2)
# output_rnn_t = tf.transpose(output_concat, perm=[1, 0, 2]) # 가장 마지막 t에서의 output값만 얻기 위해 transpose 해줌
# last_outputs = tf.gather(output_rnn_t, int(output_rnn_t.get_shape()[0]) - 1) # 가장 마지막 t에서의 output값만 빼줌

##### stacked bi-directional RNN
# 1-level layer
cell_forward1 = tf.nn.rnn_cell.LSTMCell(lstm_units)
cell_backward1 = tf.nn.rnn_cell.LSTMCell(lstm_units)
output_rnn, state_rnn = tf.nn.bidirectional_dynamic_rnn(cell_forward1, cell_backward1, x, dtype=tf.float32, scope="b1")
output_concat = tf.concat(output_rnn, axis=2)
# 2-level layer
cell_forward2 = tf.nn.rnn_cell.LSTMCell(lstm_units)
cell_backward2 = tf.nn.rnn_cell.LSTMCell(lstm_units)
output_rnn2, state_rnn2 = tf.nn.bidirectional_dynamic_rnn(cell_forward2, cell_backward2, output_concat, dtype=tf.float32, scope="b2")
output_concat2 = tf.concat(output_rnn2, axis=2)
output_rnn_t = tf.transpose(output_concat2, perm=[1, 0, 2]) # 가장 마지막 t에서의 output값만 얻기 위해 transpose 해줌
last_outputs = tf.gather(output_rnn_t, int(output_rnn_t.get_shape()[0]) - 1) # 가장 마지막 t에서의 output값만 빼줌

# # bi-directional RNN용 분류기용 가중치 매트릭스
w = tf.Variable(tf.truncated_normal(shape=[lstm_units * 2, class_count], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[class_count]))


# 예측 결과
logit = tf.matmul(last_outputs, w) + b

# 학습 영역
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

# 평가 영역
correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


################################################################################# 전처리 함수
# matrix의 size를 max_count * 300으로 바꿔주는 함수
def reshape_input_matrix(tensor, max_count):
    out_list = []

    for matrix in tensor:
        row_cnt = matrix.shape[0]
        zeros = np.zeros(((max_count - row_cnt) * 300, ))
        target = np.concatenate([zeros, matrix.reshape(-1, )])

        out_list.append(target.reshape((max_count, 300)))

    return np.array(out_list)

# 답변 벡터 만드는 함수
def reshape_answer(answer_list):
    out_list = []

    for answer in answer_list:
        if answer == 0:
            out_list.append([1, 0])
        else:
            out_list.append([0, 1])
    return out_list


################################################################################# 데이터 불러오기
with open("../dataset/sentence_matrix_train.pkl", "rb") as f:
    corpus = pickle.load(f)

with open("../dataset/answer_train.pkl", "rb") as f:
    answers = pickle.load(f)

# 훈련용, 테스트용 데이터 분리
cutting_point = 120000
train_x = corpus[:cutting_point]
train_y = answers[:cutting_point]
test_x = corpus[cutting_point:]
test_y = answers[cutting_point:]


################################################################################## 실행 코드
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    BATCH_SIZE = 100
    cnt = 0

    for idx in range(0, len(train_x), BATCH_SIZE):
        input_matrix = train_x[idx:idx + BATCH_SIZE]
        input_answer = train_y[idx:idx + BATCH_SIZE]

        x_ = reshape_input_matrix(input_matrix, max_word_count)
        y_ = reshape_answer(input_answer)

        # a = sess.run(output_rnn, feed_dict={x: x_, y: y_})
        # b = sess.run(state_rnn, feed_dict={x: x_, y: y_})
        # d = sess.run(output_concat, feed_dict={x: x_, y: y_})
        # c = sess.run(output_rnn_t, feed_dict={x: x_, y: y_})
        #
        # print("a")

        # 매 1,000개마다 100개를 가지고 성능 테스트
        if cnt % 10 == 0:
            train_accracy = sess.run(accuracy, feed_dict={x: x_, y: y_})
            print("step %d, training accuracy %g" % (idx, train_accracy))

        sess.run(optimizer, feed_dict={x: x_, y: y_})

        cnt += 1

    # 테스트
    test_x1 = reshape_input_matrix(test_x[:10000], max_word_count)
    test_y1 = reshape_answer(test_y[:10000])
    print("test accuracy %g" % sess.run(accuracy, feed_dict={x: test_x1, y: test_y1}))

