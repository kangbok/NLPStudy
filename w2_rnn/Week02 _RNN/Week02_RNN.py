import pickle
import time
import sys
import csv

# Third-party packages
import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec

# local module in the same directory
from BatchGenerator import BatchGenerator


train_morph_file_path = "../data/corpus_train.pkl"
test_morph_file_path = "../data/corpus_test.pkl"

train_ans_file_path = "../data/ratings_train.txt"
test_ans_file_path = "../data/ratings_test.txt"

w2v_gensim_model_file_path = "../data/wordvector_w2v.model"


def get_data_from_pickle(pickle_file_name):
    time_start = time.time()
    sys.stdout.write("Loading Data from \"%s\" ... " % pickle_file_name)
    with open(pickle_file_name, 'rb') as f:
        docs = pickle.load(f, encoding='utf8')
    time_end = time.time()
    sys.stdout.write("Done (%d seconds)\n" % int(time_end - time_start))

    return docs


def get_data_from_tsv(tsv_file_name):
    time_start = time.time()
    sys.stdout.write("Loading Data from \"%s\" ... " % tsv_file_name)
    with open(tsv_file_name, 'r', encoding='utf8') as f:
        docs = csv.reader(f, delimiter='\t')
        docs = list(docs)
        docs = docs[1:]
        docs = [label for _, _, label in docs]

    time_end = time.time()
    sys.stdout.write("Done (%d seconds)\n" % int(time_end - time_start))

    return docs

def get_sents_words_index(sents_words, wv_index2word):
    sents_words_index = list()  # list of list of int
    time_start = time.time()
    for i, sent_words in enumerate(sents_words):
        words_index = [wv_index2word.index(word) for word in sent_words if word in wv_index2word]
        sents_words_index.append(words_index)
        sys.stdout.write("\r[%d/%d] (%d secs)" % (i + 1, len(sents_words), (int)(time.time() - time_start)))
    print()

    return sents_words_index

def get_sents_words_vector(sents_words_index, wv_vectors, max_seq):
    time_start = time.time()
    sents_words_vector = list()
    for i, sent_words_index in enumerate(sents_words_index):
        cut_index = sent_words_index[:max_seq]
        sent_words_vector = wv_vectors[cut_index]
        if sent_words_vector.shape[0] < max_seq:
            sent_words_vector = np.pad(sent_words_vector,
                                       ((0, max_seq - sent_words_vector.shape[0]), (0, 0)), 'constant')

        # sent_words_vector = sent_words_vector.reshape((1, n_step, n_wv_dim))
        # train_sents_words_vector = np.concatenate((train_sents_words_vector, sent_words_vector), axis=0)
        sents_words_vector.append(sent_words_vector)
        sys.stdout.write("\r[%d/%d] (%d secs)" %
                         (i + 1, len(sents_words_index), (int)(time.time() - time_start)))
    print()
    sents_words_vector = np.array(sents_words_vector, dtype=np.float32)

    # print(sents_words_vector.shape)

    return sents_words_vector


if __name__ == "__main__":
    # load train data
    train_morph_data = get_data_from_pickle(train_morph_file_path)
    print(len(train_morph_data))
    print("See a Train Data Sample")
    print(train_morph_data[0])
    print()

    train_ans_data = get_data_from_tsv(train_ans_file_path)
    train_ans_data = [int(ans) for ans in train_ans_data]
    print(len(train_ans_data))
    print("See a Train Answer Sample")
    print(type(train_ans_data[0]))
    print(train_ans_data[0])
    print()

    # load gensim model
    gensim_model = Word2Vec.load(w2v_gensim_model_file_path)

    wv_words = gensim_model.wv.index2word  # list of str
    wv_vectors = gensim_model.wv.vectors  # ndarray(15409, 300)

    sys.stdout.write("Gensim \'words\' type - ")
    print(type(wv_words))
    sys.stdout.write("Gensim \'words\' length - ")
    print(len(wv_words))
    sys.stdout.write("Gensim \'words[0]\' type - ")
    print(type(wv_words[0]))
    sys.stdout.write("Gensim \'words[0]\' - ")
    print(wv_words[0])
    print()

    sys.stdout.write("Gensim \'vectors\' type - ")
    print(type(wv_vectors))
    sys.stdout.write("Gensim \'vectors\' shape - ")
    print(wv_vectors.shape)
    sys.stdout.write("Gensim \'vectors[0]\' type - ")
    print(type(wv_vectors[0]))
    sys.stdout.write("Gensim \'vectors[0]\' shape - ")
    print(wv_vectors[0].shape)
    sys.stdout.write("Gensim \'vectors[0][0]\' type - ")
    print(type(wv_vectors[0][0]))
    print()

    # settings
    n_wv_dim = 300
    n_step = 50
    n_class = 2
    n_hidden = 300
    n_stack = 2
    n_batch_train = 1000
    learning_rate = 1e-4
    n_epoch_train = 5

    # train_morph_data = train_morph_data[:100]
    # train_ans_data = train_ans_data[:100]

    # prepare data as vector
    print("Finding indexes of words in sentences in train data ...")
    train_sents_words_index = get_sents_words_index(train_morph_data, wv_words)

    print("Making matrices of word vectors in sentences in train data ...")
    train_sents_words_vector = get_sents_words_vector(train_sents_words_index, wv_vectors, n_step)

    ## make answer vector (one hot)
    train_ans_onehot = np.eye(n_class)[train_ans_data]

    ## Construct batch generator
    train_batch_generator = BatchGenerator(train_sents_words_vector, train_ans_onehot, n_batch_train)


    # Draw graph

    ## input, output
    x = tf.placeholder(tf.float32, [None, n_step, n_wv_dim])
    y = tf.placeholder(tf.float32, [None, n_class])

    ## RNN
    cells = [tf.nn.rnn_cell.LSTMCell(n_hidden) for _ in range(n_stack)]
    stacked_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    outputs_rnn, states_rnn = tf.nn.dynamic_rnn(stacked_cell, x, dtype=tf.float32)

    ## In each sentence, the hidden vector of the final step is only needed.
    ## (batch_size, n_step, n_hidden]) -> [n_step, batch_size, n_hidden]
    outputs_rnn = tf.transpose(outputs_rnn, [1, 0, 2])
    outputs_rnn = outputs_rnn[-1]

    ## Full-connected layer
    W = tf.Variable(tf.truncated_normal([n_hidden, n_class]))
    b = tf.Variable(tf.truncated_normal([n_class]))

    logit = tf.matmul(outputs_rnn, W) + b

    ## Get cost and define optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    ## Evaluation
    correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(y, 1))
    num_correct_pred = tf.reduce_sum(tf.cast(correct_prediction, "float"))

    # Run
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    time_start = time.time()
    cost_epoch = 0
    while train_batch_generator.get_epoch() < n_epoch_train:
        batch_x, batch_y = train_batch_generator.next_batch()
        _, cost_batch = sess.run([optimizer, cost],
                               feed_dict={x: batch_x, y: batch_y})
        cost_epoch += cost_batch
        if train_batch_generator.get_epoch_end():
            print('Epoch:', '%02d' % (train_batch_generator.get_epoch()),
                  '  cost =', '{:.10f}'.format(cost_epoch / len(train_morph_data)),
                  "(%d secs)" % ((int)(time.time()-time_start)))
            cost_epoch = 0

    print("Training End")

    # delete train data
    del train_morph_data
    del train_ans_data
    del train_batch_generator

    # Test
    ## load test data
    test_morph_data = get_data_from_pickle(test_morph_file_path)
    test_ans_data = get_data_from_tsv(test_ans_file_path)
    test_ans_data = [int(ans) for ans in test_ans_data]

    ## prepare test data as vector
    print("Finding indexes of words in sentences in test data ...")
    test_sents_words_index = get_sents_words_index(test_morph_data, wv_words)

    print("Making matrices of word vectors in sentences in test data ...")
    test_sents_words_vector = get_sents_words_vector(test_sents_words_index, wv_vectors, n_step)

    ## make answer vector (one hot)
    test_ans_onehot = np.eye(n_class)[test_ans_data]


    ## Construct batch generator
    n_batch_test = 2000
    test_batch_generator = BatchGenerator(test_sents_words_vector, test_ans_onehot, n_batch_test)
    num_total_correct = 0
    while test_batch_generator.get_epoch() < 1:
        batch_x, batch_y = test_batch_generator.next_batch()
        num_total_correct += sess.run(num_correct_pred, feed_dict={x: batch_x, y: batch_y})

    accuracy = num_total_correct / len(test_morph_data)
    print("Test Accuracy: %04f" % accuracy)


