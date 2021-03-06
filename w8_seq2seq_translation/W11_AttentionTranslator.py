#
# tensorflow의 contrib.seq2seq 패키지의 attention을 이용하여 구현한 seq2seq.
#

import os
import pickle

import numpy as np
import tensorflow as tf
import tensorflow.contrib as contrib

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
# LEARNING_RATE = 0.0001
BATCH_SIZE = 100


##### place holders
encoder_x = tf.placeholder(tf.int32, [None, ENCODER_MAX_LENGTH], name="encoder_x")
decoder_x = tf.placeholder(tf.int32, [None, DECODER_MAX_LENGTH], name="decoder_x")
decoder_y = tf.placeholder(tf.int32, [None, DECODER_MAX_LENGTH], name="decoder_y")
real_encoder_length = tf.placeholder(tf.int32, [None, ], name="real_encoder_length")
real_decoder_length = tf.placeholder(tf.int32, [None, ], name="real_decoder_length")
learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")

# 0.0 ≤ sampling_probability ≤ 1.0
# 0.0: no sampling => `ScheduledEmbedidngTrainingHelper` is equivalent to `TrainingHelper`
# 1.0: always sampling => `ScheduledEmbedidngTrainingHelper` is equivalent to `GreedyEmbeddingHelper`
# Inceasing sampling over steps => Curriculum Learning
# sampling_probability = tf.placeholder(tf.float32, shape=[], name="sampling_probability")

##### encoder
encoder_emb_w = tf.get_variable(name='enc_embedding', initializer=tf.random_uniform([ENCODER_VOCAB_SIZE, ENCODER_EMB_SIZE]), dtype=tf.float32)
enc_emb_inputs = tf.nn.embedding_lookup(encoder_emb_w, encoder_x, name='emb_inputs')

encoder_cell = tf.nn.rnn_cell.LSTMCell(RNN_UNITS, name="encoder_cell")
encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell, enc_emb_inputs, sequence_length=real_encoder_length, dtype=tf.float32, scope="encoder_rnn")

##### decoder(seq2seq)
decoder_emb_w = tf.get_variable(name='dec_embedding', initializer=tf.random_uniform([DECODER_VOCAB_SIZE, DECODER_EMB_SIZE]), dtype=tf.float32)
decoder_cell = tf.nn.rnn_cell.LSTMCell(RNN_UNITS, name="decoder_cell")
attention_mech = contrib.seq2seq.LuongAttention(num_units=ATTENTION_SIZE, memory=encoder_output, memory_sequence_length=real_encoder_length, name="attention_mech")
decoder_cell = contrib.seq2seq.AttentionWrapper(cell=decoder_cell, attention_mechanism=attention_mech, attention_layer_size=ATTENTION_SIZE)
initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=BATCH_SIZE).clone(cell_state=encoder_state)
output_layer = tf.layers.Dense(DECODER_VOCAB_SIZE, name='output_projection')

### training part
training_max_length = tf.reduce_max(real_decoder_length + 1, name="decoder_max_length")
dec_emb_inputs = tf.nn.embedding_lookup(decoder_emb_w, decoder_x, name='emb_inputs')
training_helper = contrib.seq2seq.TrainingHelper(inputs=dec_emb_inputs, sequence_length=real_decoder_length + 1, name="training_helper")
# training_helper = contrib.seq2seq.ScheduledEmbeddingTrainingHelper(inputs=dec_emb_inputs,
#                                                                    sequence_length=real_decoder_length + 1,
#                                                                    embedding=decoder_emb_w,
#                                                                    sampling_probability=sampling_probability,
#                                                                    name="training_helper")
training_decoder = contrib.seq2seq.BasicDecoder(decoder_cell, training_helper, initial_state, output_layer)

decoder_train_output, decoder_state_train, decoder_train_output_length = \
    contrib.seq2seq.dynamic_decode(decoder=training_decoder, maximum_iterations=training_max_length, impute_finished=True)

decoder_train_logits = tf.identity(decoder_train_output.rnn_output, name="decoder_logits_train")
targets = tf.slice(decoder_y, [0, 0], [-1, training_max_length], name="targets")
mask = tf.sequence_mask(real_decoder_length + 1, training_max_length, tf.float32, name="mask")
batch_loss = contrib.seq2seq.sequence_loss(logits=decoder_train_logits, targets=targets, weights=mask, name="batch_loss")
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(batch_loss)
valid_predictions = tf.identity(decoder_train_output.sample_id, name='valid_preds')

### inference part
batch_size = tf.shape(encoder_x)[0]
start_tokens = tf.zeros(batch_size, tf.int32, name="start_tokens")
inference_helper = contrib.seq2seq.GreedyEmbeddingHelper(embedding=decoder_emb_w,
                                                         start_tokens=start_tokens,
                                                         end_token=1)
inference_decoder = contrib.seq2seq.BasicDecoder(decoder_cell, inference_helper, initial_state, output_layer)
decoder_output_inf, decoder_state_inf, \
decoder_output_length_inf = contrib.seq2seq.dynamic_decode(decoder=inference_decoder,
                                                           maximum_iterations=DECODER_MAX_LENGTH,
                                                           impute_finished=True)
predictions = tf.identity(decoder_output_inf.sample_id, name='predictions')


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

################################################################################# 모델 저장
saver = tf.train.Saver()
SAVER_DIR = "model"
checkpoint_path = os.path.join(SAVER_DIR, "seq2seq_attention_translation")
ckpt = tf.train.get_checkpoint_state(SAVER_DIR)


################################################################################# 데이터 불러오기
with open("dataset/dataset_word_kor.pkl", "rb") as f:
    encoder_corpus = pickle.load(f)
    # encoder_corpus = encoder_corpus[100]
with open("dataset/dataset_word_eng.pkl", "rb") as f:
    decoder_corpus = pickle.load(f)
    # decoder_corpus = decoder_corpus[100]


def print_tmp_result(original_answer, predicted_result):
    predicted_result = predicted_result.reshape(1, -1)

    output_list = [idx_vocab_eng_dict[int(i)] for i in predicted_result[0]]
    print("%s" % (output_list))

    with open("translation_result.txt", "a") as f:
        f.write("%s - %s\n" % (original_answer, output_list))


################################################################################# 실행
# 훈련용, 테스트용 데이터 분리 (추후 데이터 더 생기면 분리)
# cutting_point = 150000
# encoder_xx = encoder_corpus
# decoder_xx = decoder_corpus

EPOCH = 1000
LEARNING_RATE = 0.001
# sampling_probability_list = np.linspace(start=0.0, stop=1.0, num=EPOCH, dtype=np.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    cnt = 0

    for epoch in range(EPOCH):
        for idx in range(0, len(encoder_corpus), BATCH_SIZE):
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

            ### 그냥 TrainingHelper를 사용할 시
            sess.run(optimizer, feed_dict={encoder_x: encoder_x_, real_encoder_length: encoder_size_,
                                           decoder_x: decoder_x_, real_decoder_length: decoder_size_,
                                           decoder_y: decoder_y_, learning_rate: LEARNING_RATE})

            ### ScheduledEmbeddingTrainingHelper를 사용할 시
            # sess.run(optimizer, feed_dict={encoder_x: encoder_x_, real_encoder_length: encoder_size_,
            #                                decoder_x: decoder_x_, real_decoder_length: decoder_size_,
            #                                sampling_probability: sampling_probability_list[epoch]})

            cnt += 1
