##### https://github.com/j-min/tf_tutorial_plus/blob/master/RNN_seq2seq/contrib_seq2seq/00_basic_decoder.ipynb
##### code from above URL.

import numpy as np
import tensorflow as tf
import tensorflow.contrib as contrib
from pprint import pprint


sequence_length = [3, 4, 3, 1, 0]
batch_size = 5
max_time = 8
input_size = 7
hidden_size = 10
output_size = 3

inputs = np.random.randn(batch_size, max_time, input_size).astype(np.float32)
output_layer = tf.layers.Dense(output_size) # will get a trainable variable size [hidden_size x output_size]

##### decoder
decoder_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
helper = contrib.seq2seq.TrainingHelper(inputs, sequence_length)
decoder = contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                       helper=helper,
                                       initial_state=decoder_cell.zero_state(dtype=tf.float32, batch_size=batch_size),
                                       output_layer=output_layer)

print(decoder.output_size)
print(decoder.output_dtype)
print(decoder.batch_size)

### initialize states
first_finished, first_inputs, first_state = decoder.initialize()

### Unroll single step
step_outputs, step_state, step_next_inputs, step_finished = decoder.step(tf.constant(0), first_inputs, first_state)


##### run
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    results = sess.run({
        "batch_size": decoder.batch_size,
        "first_finished": first_finished,
        "first_inputs": first_inputs,
        "first_state": first_state,
        "step_outputs": step_outputs,
        "step_state": step_state,
        "step_next_inputs": step_next_inputs,
        "step_finished": step_finished})
pprint(results)
