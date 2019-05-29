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

def indexesToInputOneHots(morph_indexes, ans, num_morph):
    oneHotNAnsList = list()
    for index in morph_indexes:
        oneHot = np.zeros((num_morph+1, ), dtype=int)
        oneHot[index] = 1
        if ans == 1:
            oneHot[-1] = 1
        oneHotNAnsList.append(oneHot)

    return np.array(oneHotNAnsList)

def batchIndexesToInputOneHots(bacthIndexesNAns, num_morph):
    seqInputList = \
        [indexesToInputOneHots(indexes, ans, num_morph) for indexes, ans in bacthIndexesNAns]
    return np.array(seqInputList)

def indexesToOutputOneHots(morph_indexes, num_morph):
    oneHotList = list()
    for index in morph_indexes:
        oneHot = np.zeros((num_morph,), dtype=int)
        oneHot[index] = 1
        oneHotList.append(oneHot)
    return np.array(oneHotList)

def batchIndexesToOutputOneHots(bacthIndexes, num_morph):
    seqOutputList = [indexesToOutputOneHots(indexes, num_morph) for indexes in bacthIndexes]
    return np.array(seqOutputList)


if __name__ == "__main__":
    # load train data
    trainMorphData = get_data_from_pickle(train_morph_file_path)
    print(len(trainMorphData))
    print("See a Train Data Sample")
    print(trainMorphData[0])
    print()

    # trainMorphData = trainMorphData[:10020]

    trainAnsData = get_data_from_tsv(train_ans_file_path)
    trainAnsData = [int(ans) for ans in trainAnsData]
    print(len(trainAnsData))
    print("See a Train Answer Sample")
    print(type(trainAnsData[0]))
    print(trainAnsData[0])
    print()

    # load gensim model
    gensim_model = Word2Vec.load(w2v_gensim_model_file_path)
    morphs = gensim_model.wv.index2word  # list of str
    morphToIndex = dict()
    for i, word in enumerate(morphs):
        morphToIndex[word] = i

    # hyper parameters
    seqLen = 10
    numBatchTrain = 100
    vecInputDim = None
    vecOutputDim = None
    numHidden = 256
    numStack = 2
    learningRate = 1e-3
    numEpochTrain = 5

    startSeq = ["<BOS-{}>".format(i) for i in range(seqLen-1)]
    startSeq.reverse()
    morphs.extend(startSeq)

    endMark = "<EOS>"
    morphs.append(endMark)

    morphToIndex = dict()
    for i, morph in enumerate(morphs):
        morphToIndex[morph] = i

    print("Collecting frequent morphs ...")
    time_start = time.time()
    trainMorphDataIndex = list()
    for i, sentMorph in enumerate(trainMorphData):
        augSentMorph = copy.deepcopy(startSeq)
        augSentMorphIndex = list()
        for start in startSeq:
            augSentMorphIndex.append(morphToIndex[start])

        for jMorph in sentMorph:
            if jMorph in morphs:
                augSentMorph.append(jMorph)
                augSentMorphIndex.append(morphToIndex[jMorph])

        augSentMorph.append(endMark)
        augSentMorphIndex.append(morphToIndex[endMark])

        trainMorphData[i] = augSentMorph
        trainMorphDataIndex.append(augSentMorphIndex)

        sys.stdout.write("\r[%d/%d] (%d secs)" %
                         (i + 1, len(trainMorphData), (int)(time.time() - time_start)))

    sys.stdout.write("\n")

    trainSentIndexNAnsInputRnn = list()
    trainSentIndexOutputRnn = list()


    for aSentIndex, ans in zip(trainMorphDataIndex, trainAnsData):
        for jPart in range(len(aSentIndex) - seqLen):
            inputIndex = aSentIndex[jPart:jPart+seqLen]
            trainSentIndexNAnsInputRnn.append((inputIndex, ans))
            outputIndex = aSentIndex[jPart+1:jPart+seqLen+1]
            trainSentIndexOutputRnn.append(outputIndex)
            # print(len(outputIndex))

    print("trainSentIndexNAnsInputRnn length - %d" % len(trainSentIndexNAnsInputRnn))
    print("one of trainSentIndexNAnsInputRnn length - %d" % len(trainSentIndexNAnsInputRnn[0][0]))
    print("trainSentIndexOutputRnn length - %d" % len(trainSentIndexOutputRnn))
    print("one of trainSentIndexOutputRnn length - %d" % len(trainSentIndexOutputRnn[0]))

    numMorph = len(morphs)
    vecInputDim = numMorph + 1
    vecOutputDim = numMorph

    ## Construct batch generator
    train_batch_generator = BatchGenerator(trainSentIndexNAnsInputRnn, trainSentIndexOutputRnn, numBatchTrain)
    print("BatchGenerator is constructed.")

    # Draw graph

    ## input, output
    x = tf.placeholder(tf.float32, [None, seqLen, vecInputDim])
    y = tf.placeholder(tf.float32, [None, seqLen, vecOutputDim])

    ## RNN
    # cells = [tf.nn.rnn_cell.LSTMCell(numHidden) for _ in range(numStack)]
    # stackedCell = tf.nn.rnn_cell.MultiRNNCell(cells)
    # outputsRnn, statesRnn = tf.nn.dynamic_rnn(stackedCell, x, dtype=tf.float32)

    outputsRnn, statesRnn = tf.nn.dynamic_rnn(tf.nn.rnn_cell.LSTMCell(numHidden), x, dtype=tf.float32)

    ## In each sentence, the hidden vector of the final step is only needed.
    ## (batch_size, n_step, n_hidden]) -> [n_step, batch_size, n_hidden]
    # outputsRnn = tf.transpose(outputsRnn, [1, 0, 2])
    outputsRnn = tf.reshape(outputsRnn, [tf.shape(outputsRnn)[0], -1])
    # lastOutputRnn = outputsRnn[-1]

    ## Full-connected layer
    W = tf.Variable(tf.truncated_normal([numHidden*seqLen, vecOutputDim]))
    b = tf.Variable(tf.truncated_normal([vecOutputDim]))

    logit = tf.matmul(outputsRnn, W) + b

    yLast = tf.transpose(y, [1, 0, 2])[-1]

    ## Get cost and define optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=yLast))
    optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost)


    print("Graph is drawn and session starts")

    # Run
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # print("tf.global_variables_initializer()")
    time_start = time.time()
    cost_epoch = 0
    while train_batch_generator.get_epoch() < numEpochTrain:
        batchIndexesX, batchIndexesY = train_batch_generator.next_batch()
        batchX = batchIndexesToInputOneHots(batchIndexesX, numMorph)
        batchY = batchIndexesToOutputOneHots(batchIndexesY, numMorph)
        _, cost_batch = sess.run([optimizer, cost],
                                 feed_dict={x: batchX, y: batchY})
        cost_epoch += cost_batch
        if True:
        # if train_batch_generator.get_epoch_end():
            print('Epoch:', '%02d' % (train_batch_generator.get_epoch()),
                  '  cost =', '{:.10f}'.format(cost_epoch / len(trainMorphData)),
                  "(%d secs)" % ((int)(time.time() - time_start)))
            cost_epoch = 0

    print("Training End")


















    # print("Making one hots Raw ...")
    # time_start = time.time()
    # trainSentOneHotInputRaw = list()  # will change to np.ndarray
    # trainSentOneHotOutputRaw = list()  # will change to np.ndarray
    #
    # for i, sentMorphIdx in enumerate(trainMorphDataIndex):
    #
    #     sentInputOneHot = indexesToInputOneHots(sentMorphIdx, numMorph, trainAnsData[i])
    #     trainSentOneHotInputRaw.append(sentInputOneHot)
    #
    #     sentOutputOneHot = indexesToOutputOneHots(sentMorphIdx, numMorph)
    #     trainSentOneHotOutputRaw.append(sentOutputOneHot)
    #
    #     sys.stdout.write("\r[%d/%d] (%d secs)" %
    #                      (i + 1, len(trainMorphDataIndex), (int)(time.time() - time_start)))
    #
    # sys.stdout.write("\n")
    #
    # trainSentOneHotInputRaw = np.array(trainSentOneHotInputRaw)
    # trainSentOneHotOutputRaw = np.array(trainSentOneHotOutputRaw)
    #
    # trainSentOneHotInputRnn = list()
    # trainSentOneHotOutputRnn = list()
    #
    # for iSent in range(len(trainSentOneHotInputRaw)):
    #     oneSentInputRaw = trainSentOneHotInputRaw[iSent]
    #     oneSentOutputRaw = trainSentOneHotOutputRaw[iSent]
    #     for jPart in range(oneSentInputRaw.shape(0) - seqLen):
    #         input = oneSentInputRaw[jPart:jPart+seqLen]
    #         trainSentOneHotInputRnn.append(input)
    #         output = oneSentOutputRaw[jPart+1:jPart+seqLen+1]
    #         trainSentOneHotOutputRnn.append(output)
    #
    # trainSentOneHotInputRnn = np.array(trainSentOneHotInputRnn, dtype=np.float32)
    # trainSentOneHotOutputRnn = np.array(trainSentOneHotOutputRnn, dtype=np.float32)
    





