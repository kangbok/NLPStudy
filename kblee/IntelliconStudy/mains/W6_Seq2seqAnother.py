import os
import pickle

import numpy as np
import tensorflow as tf
import tensorflow.contrib as contrib

with open("../resource/vocab_idx_dict.pkl", "rb") as f:
    vocab_idx_dict = pickle.load(f)

with open("../resource/idx_vocab_dict.pkl", "rb") as f:
    idx_vocab_dict = pickle.load(f)

batch_size = 1
vocab_size = len(vocab_idx_dict.keys())
encoder_lstm_units = 256
decoder_lstm_units = 256
decoder_max_length = tf.constant(100)

encoder_x = tf.placeholder(tf.float32, [1, None, vocab_size])
decoder_x = tf.placeholder(tf.float32, [1, None, vocab_size])
decoder_y = tf.placeholder(tf.float32, [1, None, vocab_size])

##### encoder
encoder_cell = tf.nn.rnn_cell.LSTMCell(encoder_lstm_units)
encoder_dropout_cell = tf.nn.rnn_cell.DropoutWrapper(encoder_cell, output_keep_prob=0.75)
encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_dropout_cell, encoder_x, dtype=tf.float32, scope="encoder_rnn")


##### decoder (seq2seq)
decoder_length = tf.shape(decoder_x)[0]
decoder_cell = tf.nn.rnn_cell.LSTMCell(decoder_lstm_units)
training_helper = contrib.seq2seq.TrainingHelper(inputs=decoder_x, sequence_length=decoder_length)
training_decoder = contrib.seq2seq.BasicDecoder(decoder_cell, training_helper, encoder_state)

decoder_output_train, decoder_state_train, \
decoder_output_length_train = contrib.seq2seq.dynamic_decode(decoder=training_decoder, maximum_iterations=decoder_max_length)

decoder_logits_train = tf.identity(decoder_output_train)
decoder_prediction_train = tf.argmax(decoder_logits_train, axis=-1)
mask = tf.sequence_mask(decoder_length, decoder_max_length, tf.float32)
loss = contrib.seq2seq.sequence_loss(logits=decoder_logits_train, targets=decoder_prediction_train)


