#
# W11_AttentionTranslator.py에서 만든 모델 파일이 있을 경우, 해당 파일을 불러와서 새로운 데이터로 다시 학습 시키는 코드.
#

import os
import pickle

import numpy as np
import tensorflow as tf

##### vocab 관련 파일 로딩
with open("resource/vocab_idx_word_kor.pkl", "rb") as f:
    vocab_idx_kor_dict = pickle.load(f)
with open("resource/vocab_idx_word_eng.pkl", "rb") as f:
    vocab_idx_eng_dict = pickle.load(f)
with open("resource/idx_vocab_word_kor.pkl", "rb") as f:
    idx_vocab_kor_dict = pickle.load(f)
with open("resource/idx_vocab_word_eng.pkl", "rb") as f:
    idx_vocab_eng_dict = pickle.load(f)

##### 연산에 쓰이는 상수들 정의
ENCODER_VOCAB_SIZE = len(vocab_idx_kor_dict.keys())
DECODER_VOCAB_SIZE = len(vocab_idx_eng_dict.keys())
ENCODER_EMB_SIZE = 128
DECODER_EMB_SIZE = 128
RNN_UNITS = 256
ENCODER_MAX_LENGTH = 150
DECODER_MAX_LENGTH = 50
ATTENTION_SIZE = 256
BATCH_SIZE = 100


################################################################################# 전처리 함수
# 인코더 문장을 word idx의 리스트로 바꿔줌
def create_encoder_input(stc_list, batch_size):
    out_idx_list = []
    out_encoder_size = []

    for word_list in stc_list:
        one_encoder_list = []

        for idx, word in enumerate(word_list):
            if idx >= ENCODER_MAX_LENGTH - 2:
                break

            one_encoder_list.append(vocab_idx_kor_dict[word])

        one_encoder_list = np.asarray(one_encoder_list, dtype="int32")

        for _ in range(ENCODER_MAX_LENGTH - len(one_encoder_list)):
            one_encoder_list = np.append(one_encoder_list, 2)  # 2의 의미 : <P> 토큰

        out_idx_list.append(one_encoder_list)
        out_encoder_size.append(min(len(word_list), ENCODER_MAX_LENGTH - 1))

    out_idx_list = np.asarray(out_idx_list, dtype=np.int)
    out_idx_list = out_idx_list.reshape([batch_size, -1])

    try:
        out_encoder_size = np.asarray(out_encoder_size, dtype=np.int).reshape((batch_size,))
    except ValueError as e:
        return None, None

    return out_idx_list, out_encoder_size


def create_decoder_input(stc_list, batch_size):
    out_x_idx_list = []
    out_y_idx_list = []
    out_decoder_size = []

    for word_list in stc_list:
        one_decoder_x_list = [0] # 0의 의미 : <S> 토큰
        one_decoder_y_list = []

        for idx, word in enumerate(word_list):
            if idx >= DECODER_MAX_LENGTH - 2:
                break

            one_decoder_x_list.append(vocab_idx_eng_dict[word])
            one_decoder_y_list.append(vocab_idx_eng_dict[word])

        one_decoder_x_list = np.asarray(one_decoder_x_list, dtype=np.int)
        one_decoder_y_list = np.asarray(one_decoder_y_list, dtype=np.int)

        for _ in range(DECODER_MAX_LENGTH - len(one_decoder_x_list)):
            one_decoder_x_list = np.append(one_decoder_x_list, 1)  # 1의 의미 : <E> 토큰
        for _ in range(DECODER_MAX_LENGTH - len(one_decoder_y_list)):
            one_decoder_y_list = np.append(one_decoder_y_list, 1)  # 1의 의미 : <E> 토큰


        out_x_idx_list.append(one_decoder_x_list)
        out_y_idx_list.append(one_decoder_y_list)
        out_decoder_size.append(min(len(word_list), DECODER_MAX_LENGTH - 1))

    out_x_idx_list = np.asarray(out_x_idx_list, dtype=np.int)
    out_x_idx_list = out_x_idx_list.reshape([batch_size, -1])
    out_y_idx_list = np.asarray(out_y_idx_list, dtype=np.int)
    out_y_idx_list = out_y_idx_list.reshape([batch_size, -1])

    out_decoder_size = np.asarray(out_decoder_size, dtype=np.int).reshape((batch_size, ))

    return out_x_idx_list, out_y_idx_list, out_decoder_size


################################################################################# 데이터 불러오기
with open("dataset/dataset_word_kor.pkl", "rb") as f:
    encoder_corpus = pickle.load(f)
with open("dataset/dataset_word_eng.pkl", "rb") as f:
    decoder_corpus = pickle.load(f)


def print_tmp_result(original_answer, predicted_result):
    predicted_result = predicted_result.reshape(1, -1)

    output_list = [idx_vocab_eng_dict[int(i)] for i in predicted_result[0]]
    print("%s" % (output_list))

    with open("translation_result.txt", "a") as f:
        f.write("%s - %s\n" % (original_answer, output_list))


################################################################################# 실행
EPOCH = 1000
LEARNING_RATE = 0.0001

with tf.Session() as sess:
    loader = tf.train.import_meta_graph("model/seq2seq_attention_translation.meta")
    loader.restore(sess, tf.train.latest_checkpoint("model/"))

    graph = tf.get_default_graph()
    batch_loss = graph.get_tensor_by_name("batch_loss/Sum:0")
    optimizer = graph.get_operation_by_name("Adam")
    predictions = graph.get_tensor_by_name("predictions:0")
    encoder_x = graph.get_tensor_by_name("encoder_x:0")
    real_encoder_length = graph.get_tensor_by_name("real_encoder_length:0")
    decoder_x = graph.get_tensor_by_name("decoder_x:0")
    decoder_y = graph.get_tensor_by_name("decoder_y:0")
    real_decoder_length = graph.get_tensor_by_name("real_decoder_length:0")
    learning_rate = graph.get_tensor_by_name("learning_rate:0")

    ################################################################################# 모델 저장 관련 변수 선언
    saver = tf.train.Saver()
    SAVER_DIR = "model"
    checkpoint_path = os.path.join(SAVER_DIR, "seq2seq_attention_translation")
    ckpt = tf.train.get_checkpoint_state(SAVER_DIR)

    ################################################################################# 돌아가면서 학습 시작
    cnt = 0
    is_start = False

    for epoch in range(EPOCH):
        for idx in range(0, len(encoder_corpus), BATCH_SIZE):
            if cnt > 1350600:
                is_start = True

            if not is_start:
                cnt += 1
                continue

            input_words = encoder_corpus[idx:idx + BATCH_SIZE]
            output_words = decoder_corpus[idx:idx + BATCH_SIZE]

            encoder_x_, encoder_size_ = create_encoder_input(input_words, BATCH_SIZE)

            if encoder_x_ is None or encoder_x_.shape[0] < BATCH_SIZE:
                continue

            decoder_x_, decoder_y_, decoder_size_ = create_decoder_input(output_words, BATCH_SIZE)

            # 매 100개 batch마다 step print, 모델 저장
            if cnt % 100 == 0:
                loss = sess.run(batch_loss, feed_dict={encoder_x: encoder_x_, real_encoder_length: encoder_size_,
                                                       decoder_x: decoder_x_, real_decoder_length: decoder_size_,
                                                       decoder_y: decoder_y_})
                test_result = sess.run(predictions, feed_dict={encoder_x: encoder_x_, real_encoder_length: encoder_size_})

                print("step %s  /  loss %s" % (cnt, loss))
                print(input_words[0])
                print(output_words[0])
                print_tmp_result(output_words[0], test_result[0])
                saver.save(sess, checkpoint_path)

            sess.run(optimizer, feed_dict={encoder_x: encoder_x_, real_encoder_length: encoder_size_,
                                           decoder_x: decoder_x_, real_decoder_length: decoder_size_,
                                           decoder_y: decoder_y_, learning_rate: LEARNING_RATE})

            cnt += 1
