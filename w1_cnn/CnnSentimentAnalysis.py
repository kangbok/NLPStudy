import pickle

import numpy as np
import tensorflow as tf

###################################################################### 관련 함수 선언
# 가중치 메트릭스 초기화 함수
def initialize_weight_matrix(shape):
    initial = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# 바이어스항 초기화 함수
def initialize_bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

###################################################################### CNN 시작
x = tf.placeholder("float", [None, 100, 300])
y = tf.placeholder("float", [None, 2])

x_ = tf.reshape(x, [-1, 100, 300, 1])

# convolution kernel들 정의
kernel1 = initialize_weight_matrix([1, 300, 1, 10]) # 1 * 300짜리 필터 10개
kernel2 = initialize_weight_matrix([2, 300, 1, 10]) # 2 * 300짜리 필터 10개
kernel3 = initialize_weight_matrix([3, 300, 1, 10]) # 3 * 300짜리 필터 10개
kernel4 = initialize_weight_matrix([4, 300, 1, 10]) # 4 * 300짜리 필터 10개
kernel5 = initialize_weight_matrix([5, 300, 1, 10]) # 5 * 300짜리 필터 10개
b1 = initialize_bias([10])
b2 = initialize_bias([10])
b3 = initialize_bias([10])
b4 = initialize_bias([10])
b5 = initialize_bias([10])

# convolution 수행
conv1 = (tf.nn.conv2d(x_, kernel1, strides=[1, 1, 1, 1], padding="VALID") + b1)
conv2 = (tf.nn.conv2d(x_, kernel2, strides=[1, 1, 1, 1], padding="VALID") + b2)
conv3 = (tf.nn.conv2d(x_, kernel3, strides=[1, 1, 1, 1], padding="VALID") + b3)
conv4 = (tf.nn.conv2d(x_, kernel4, strides=[1, 1, 1, 1], padding="VALID") + b4)
conv5 = (tf.nn.conv2d(x_, kernel5, strides=[1, 1, 1, 1], padding="VALID") + b5)

# max pooling 수행
pool1 = tf.nn.max_pool(conv1, ksize=[1, 100, 1, 1], strides=[1, 1, 1, 1], padding="VALID")
pool2 = tf.nn.max_pool(conv2, ksize=[1, 99, 1, 1], strides=[1, 1, 1, 1], padding="VALID")
pool3 = tf.nn.max_pool(conv3, ksize=[1, 98, 1, 1], strides=[1, 1, 1, 1], padding="VALID")
pool4 = tf.nn.max_pool(conv4, ksize=[1, 97, 1, 1], strides=[1, 1, 1, 1], padding="VALID")
pool5 = tf.nn.max_pool(conv5, ksize=[1, 96, 1, 1], strides=[1, 1, 1, 1], padding="VALID")

fv_part1 = tf.reshape(pool1, [-1, 10])
fv_part2 = tf.reshape(pool2, [-1, 10])
fv_part3 = tf.reshape(pool3, [-1, 10])
fv_part4 = tf.reshape(pool4, [-1, 10])
fv_part5 = tf.reshape(pool5, [-1, 10])

feature_vector = tf.concat([fv_part1, fv_part2, fv_part3, fv_part4, fv_part5], 1)

# FCN 첫 번째 레이어
fcn1_w = initialize_weight_matrix([50, 512])
fcn1_b = initialize_bias([512])
h_fcn1 = tf.nn.relu(tf.matmul(feature_vector, fcn1_w) + fcn1_b)

keep_prob = tf.placeholder("float")
h_fcn1_drop = tf.nn.dropout(h_fcn1, keep_prob) # dropout 해줌

# FCN 두 번째 레이어
fcn2_w = initialize_weight_matrix([512, 2])
fcn2_b = initialize_bias([2])

# softmax 형태로 아웃풋 뽑기
y_conv = tf.nn.softmax(tf.matmul(h_fcn1_drop, fcn2_w) + fcn2_b)

# 평가 영역
cross_entropy = -tf.reduce_sum(y * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))



###################################################################### 실행 코드
# 데이터 불러오기
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

# 전처리 함수 - matrix의 size를 100 * 300으로 바꿔주는 함수
def reshape_input_matrix(tensor):
    out_list = []

    for matrix in tensor:
        row_cnt = matrix.shape[0]
        zeros = np.zeros(((100 - row_cnt) * 300, ))
        target = np.concatenate([matrix.reshape(-1, ), zeros])

        out_list.append(target.reshape((100, 300)))

    return np.array(out_list)

def reshape_answer(answer_list):
    out_list = []

    for answer in answer_list:
        if answer == 0:
            out_list.append([1, 0])
        else:
            out_list.append([0, 1])
    return out_list

# 정말 실행 코드
sess = tf.Session()
sess.run(tf.global_variables_initializer())

cnt = 0
for idx in range(0, len(train_x), 100):
    input_matrix = train_x[idx:idx + 100]
    input_answer = train_y[idx:idx + 100]

    x_ = reshape_input_matrix(input_matrix)
    y_ = reshape_answer(input_answer)

    if cnt % 10 == 0:
        train_accracy = sess.run(accuracy, feed_dict={x: x_, y: y_, keep_prob: 1.0})
        print("step %d, training accuracy %g" % (idx, train_accracy))

    sess.run(train_step, feed_dict={x:x_, y:y_, keep_prob:0.5})

    cnt += 1

test_x1 = reshape_input_matrix(test_x[:100])
test_y1 = reshape_answer(test_y[:100])
print("test accuracy %g" % sess.run(accuracy, feed_dict={x:test_x1, y:test_y1, keep_prob:1.0}))






###################################################################### 테스트용 코드
# import random
# x_list = [random.random() for _ in range(900)]
# x_tmp = [0 for _ in range(77 * 300)]
# x_list.extend(x_tmp)
# x_test = np.array(x_list)
# # x_test = x_test.reshape([1, 80, 300])
# x_test = np.array([x_test.reshape([80, 300]), x_test.reshape([80, 300])])
# y_test = np.array([[1, 0], [1, 0]])
# y_test = y_test.reshape([2,2])
#
# # config = tf.ConfigProto()
# # config.gpu_options.allow_growth = True
# # sess = tf.Session(config=config)
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# a1 = sess.run(feature_vector, feed_dict={x:x_test, y:y_test})
#
# print("a")


