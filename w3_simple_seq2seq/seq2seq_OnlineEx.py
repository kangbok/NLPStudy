# -*- coding: utf-8 -*-

import sys
import tensorflow as tf
import numpy as np
import string

lowercase = string.ascii_lowercase[:]
# uppercase = string.ascii_uppercase[:]
uppercase = 'SEP'
hangul = '윤연구원이권님황허인텔리콘메타법판례자치규행정칙단어나무놀이소녀키스사랑'
# hangul = '단어나무놀이소녀키스사랑'

all_text = lowercase + uppercase + hangul

print(str(all_text))
char_arr = [c for c in str(all_text)]

num_dic = {n: i for i, n in enumerate(char_arr)}

dic_len = len(num_dic)

print(len(char_arr))

seq_data = [['youn', '윤연구원'], ['lee', '이연권님'], ['hwang', '황연권님'],
            ['heo', '허연권님'], ['intellicon', '인텔리콘'], ['meta', '메타'],
            ['law', '법'], ['case', '판례'], ['atnmy', '자치법규'],
            ['admn', '행정규칙'],
            ['word', '단어'], ['wood', '나무'], ['game', '놀이'],
            ['girl', '소녀'], ['kiss', '키스'], ['love', '사랑']
            ]
# seq_data = [['word', '단어'], ['wood', '나무'], ['game', '놀이'],
#             ['girl', '소녀'], ['kiss', '키스'], ['love', '사랑']
#             ]


def make_batch(seq_data):
    input_batch = []
    output_batch = []
    target_batch = []

    for seq in seq_data:
        input = [num_dic[n] for n in seq[0]]
        output = [num_dic[n] for n in ('S' + seq[1])]
        target = [num_dic[n] for n in (seq[1] + 'E')]

        input_batch.append(np.eye(dic_len)[input])
        output_batch.append(np.eye(dic_len)[output])
        target_batch.append(target)

    return input_batch, output_batch, target_batch


def translate(sess, model, word):
    seq_data = [word, 'P' * len(word)]

    input_batch, output_batch, target_batch = make_batch([seq_data])

    prediction = tf.argmax(model, 2)

    result = sess.run(prediction, feed_dict={enc_input: input_batch,
                                             dec_input: output_batch,
                                             targets: target_batch})

    decoded = [char_arr[i] for i in result[0]]

    try:
        end = decoded.index('E')
        translated = ''.join(decoded[:end])

        return translated
    except:
        return ''.join(decoded)


learning_rate = 0.01
n_hidden = 128
total_epoch = 200

n_input = n_class = dic_len

enc_input = tf.placeholder(tf.float32, [None, None, n_input])
dec_input = tf.placeholder(tf.float32, [None, None, n_input])
targets = tf.placeholder(tf.int64, [None, None])

with tf.variable_scope('encode'):
    enc_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5)

    outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input,
                                            dtype=tf.float32)

with tf.variable_scope('decode'):
    dec_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5)

    outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_input,
                                            initial_state=enc_states, dtype=tf.float32)

model = tf.layers.dense(outputs, n_class, activation=None)

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=model, labels=targets
))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('./logs', sess.graph)

    input_batch, output_batch, target_batch = make_batch(seq_data)

    for epoch in range(total_epoch):
        _, loss = sess.run([optimizer, cost],
                           feed_dict={enc_input: input_batch,
                                      dec_input: output_batch,
                                      targets: target_batch})

        print('Epoch:', '%04d' % (epoch + 1),
              'cost =', '{:.6f}'.format(loss))
        sys.stdout.flush()

    # print('youn ->', translate(sess, model, 'youn'))
    # print('intellicon ->', translate(sess, model, 'intellicon'))
    # print('mtea ->', translate(sess, model, 'mtea'))
    # print('lee ->', translate(sess, model, 'lee'))
    # print('admn ->', translate(sess, model, 'admn'))

    print('word ->', translate(sess, model, 'word'))
    print('wodr ->', translate(sess, model, 'wodr'))
    print('love ->', translate(sess, model, 'love'))
    print('loev ->', translate(sess, model, 'loev'))
    print('abcd ->', translate(sess, model, 'abcd'))