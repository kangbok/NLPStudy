import pickle
import time
import sys
import csv
import copy

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


if __name__ == "__main__":
    # load train data
    trainMorphData = get_data_from_pickle(train_morph_file_path)
    print(len(trainMorphData))
    print("See a Train Data Sample")
    print(trainMorphData[0])
    print()

    trainAnsData = get_data_from_tsv(train_ans_file_path)
    trainAnsData = [int(ans) for ans in trainAnsData]
    print(len(trainAnsData))
    print("See a Train Answer Sample")
    print(type(trainAnsData[0]))
    print(trainAnsData[0])
    print()

    # hyper parameters
    seqLen = 10
    numBatchTrain = 1000
    vecInputDim = None
    vecOutputDim = None
    numHidden = None
    numStack = 2
    learningRate = 1e-4
    numEpochTrain = 5

    startSeq = ["<BOS-{}>".format(i) for i in range(seqLen-1)]
    startSeq.reverse()

    for i, sentMorph in enumerate(trainMorphData):
        augSentMorph = copy.deepcopy(startSeq)
        augSentMorph.extend(sentMorph)
        augSentMorph.append("<EOS>")
        trainMorphData[i] = augSentMorph

    allUniqueMorphs = set()
    for sent in trainMorphData:
        allUniqueMorphs.update(sent)

    indexToMorph = list(allUniqueMorphs)
    morphToIndex = dict()
    for index, morph in enumerate(indexToMorph):
        morphToIndex[morph] = index

    vecInputDim = len(indexToMorph) + 1
    vecOutputDim = len(indexToMorph)
    numHidden = vecOutputDim

    identityMatrix = np.eye(len(indexToMorph))
    oneHotsForPositiveInput = \
        np.concatenate((identityMatrix, np.ones((len(indexToMorph), 1), dtype=int)),
                        axis=1)
    oneHotsForNegativeInput = \
        np.concatenate((identityMatrix, np.zeros((len(indexToMorph), 1), dtype=int)),
                        axis=1)

    trainMorphIdx = list()
    trainSentOneHotInputRaw = list()  # will change to np.ndarray
    trainSentOneHotOutputRaw = list()  # will change to np.ndarray

    for i, sentMorph in enumerate(trainMorphData):
        sentIdx = [morphToIndex[morph] for morph in sentMorph]
        trainMorphIdx.append(sentIdx)

        if (trainAnsData[i] > 0):
            sentInputOneHot = oneHotsForPositiveInput[sentIdx]
        else:
            sentInputOneHot = oneHotsForNegativeInput[sentIdx]
        trainSentOneHotInputRaw.append(sentInputOneHot)

        sentOutputOneHot = identityMatrix[sentIdx]
        trainSentOneHotOutputRaw.append(sentOutputOneHot)

    trainSentOneHotInputRnn = list()
    trainSentOneHotOutputRnn = list()

    for iSent in range(len(trainSentOneHotInputRaw)):
        oneSentInputRaw = trainSentOneHotInputRaw[iSent]
        oneSentOutputRaw = trainSentOneHotOutputRaw[iSent]
        for jPart in range(oneSentInputRaw.shape(0) - seqLen):
            input = oneSentInputRaw[jPart:jPart + seqLen]
            trainSentOneHotInputRnn.append(input)
            output = oneSentOutputRaw[jPart+1:jPart + seqLen + 1]
            trainSentOneHotOutputRnn.append(output)

    trainSentOneHotInputRnn = np.array(trainSentOneHotInputRnn, dtype=np.float32)
    trainSentOneHotOutputRnn = np.array(trainSentOneHotOutputRnn, dtype=np.float32)

    ## Construct batch generator
    train_batch_generator = BatchGenerator(trainSentOneHotInputRnn, trainSentOneHotOutputRnn, numBatchTrain)


    # Draw graph

    ## input, output
    x = tf.placeholder(tf.float32, [None, seqLen, vecInputDim])
    y = tf.placeholder(tf.float32, [None, seqLen, vecOutputDim])

    ## RNN
    cells = [tf.nn.rnn_cell.LSTMCell(numHidden) for _ in range(numStack)]
    stackedCell = tf.nn.rnn_cell.MultiRNNCell(cells)
    outputsRnn, statesRnn = tf.nn.dynamic_rnn(stackedCell, x, dtype=tf.float32)

    outputsRnnNor = tf.math.l2_normalize(outputsRnn, axis=2)

    cost = tf.losses.mean_squared_error(y, outputsRnnNor)
    optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost)

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
                  "(%d secs)" % ((int)(time.time() - time_start)))
            cost_epoch = 0

    print("Training End")
    





