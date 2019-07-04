import sys
import random

import numpy as np
import tensorflow as tf

from Week03_Seq2Seq import train_morph_file_path, \
    w2v_gensim_model_file_path

from Week03_Seq2Seq import trainSentIndexNAnsInputRnn, morphs, numMorph

from Week03_Seq2Seq import indexesToInputOneHots, batchIndexesToInputOneHots
from Week03_Seq2Seq import indexesToOutputOneHots, batchIndexesToOutputOneHots


def nextInfer(tf_sess, tf_inference, prevIndexes, currInfer, ans, numMorph):
    nextIndexes = prevIndexes[1:]
    nextIndexes.append(currInfer)
    indexesNAnsList = [(nextIndexes, ans)]
    inputForTF = batchIndexesToInputOneHots(indexesNAnsList, numMorph)
    nextInfer = tf_sess.run(tf_inference, feed_dict={x: inputForTF})

    return nextInfer, nextIndexes


if __name__ == "__main__":
    sess = tf.Session()

    saver = tf.train.import_meta_graph('./model/rnn_gen.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./model/'))


    graph = tf.get_default_graph()

    x = graph.get_tensor_by_name("x:0")
    inference = graph.get_tensor_by_name("inference:0")

    randomPositiveSentIndex = -1
    while randomPositiveSentIndex < 0:
        randomPositiveSentIndex = random.randrange(0, len(trainSentIndexNAnsInputRnn))
        if trainSentIndexNAnsInputRnn[randomPositiveSentIndex][0][0] != 0:
            randomPositiveSentIndex = -1
        if trainSentIndexNAnsInputRnn[randomPositiveSentIndex][1] != 1:
            randomPositiveSentIndex = -1

    randomNegativeSentIndex = -1
    while randomNegativeSentIndex < 0:
        randomNegativeSentIndex = random.randrange(0, len(trainSentIndexNAnsInputRnn))
        if trainSentIndexNAnsInputRnn[randomNegativeSentIndex][0][0] != 0:
            randomNegativeSentIndex = -1
        if trainSentIndexNAnsInputRnn[randomNegativeSentIndex][1] != 0:
            randomNegativeSentIndex = -1

    positiveInferInputIndexNAnsList = [trainSentIndexNAnsInputRnn[randomPositiveSentIndex]]
    negativeInferInputIndexNAnsList = [trainSentIndexNAnsInputRnn[randomNegativeSentIndex]]

    positiveOneX = batchIndexesToInputOneHots(positiveInferInputIndexNAnsList, numMorph)
    negativeOneX = batchIndexesToInputOneHots(negativeInferInputIndexNAnsList, numMorph)

    positiveInfer = sess.run(inference, feed_dict={x: positiveOneX})
    # positiveInfer = tf.cast(positiveInfer[0], "int")
    negativeInfer = sess.run(inference, feed_dict={x: negativeOneX})
    # negativeInfer = tf.cast(negativeInfer[0], "int")


    sys.stdout.write("Positive - ")
    # for idx in trainSentIndexNAnsInputRnn[randomPositiveSentIndex][0]:
    #     sys.stdout.write(morphs[idx] + " ")
    firstIdx = trainSentIndexNAnsInputRnn[randomPositiveSentIndex][0][-1]
    sys.stdout.write(morphs[firstIdx] + " ")
    sys.stdout.write(morphs[int(positiveInfer)] + " ")

    positiveInferInputIndexes = trainSentIndexNAnsInputRnn[randomPositiveSentIndex][0]
    # positiveInferInputIndexes = positiveInferInputIndexes[1:]
    # positiveInferInputIndexes.append(positiveInfer)
    # positiveInferInputIndexNAnsList = [(positiveInferInputIndexes, 1)]
    # positiveOneX = batchIndexesToInputOneHots(positiveInferInputIndexNAnsList, numMorph)
    # positiveInfer = sess.run(inference, feed_dict={x: positiveOneX})
    for _ in range(20):
        positiveInfer, positiveInferInputIndexes = \
            nextInfer(sess, inference, positiveInferInputIndexes, positiveInfer, 1, numMorph)
        sys.stdout.write(morphs[int(positiveInfer)] + " ")

        if morphs[int(positiveInfer)] == "<EOS>":
            break

    sys.stdout.write("\n")


    sys.stdout.write("Negative - ")
    # for idx in trainSentIndexNAnsInputRnn[randomNegativeSentIndex][0]:
    #     sys.stdout.write(morphs[idx] + " ")
    firstIdx = trainSentIndexNAnsInputRnn[randomNegativeSentIndex][0][-1]
    sys.stdout.write(morphs[int(negativeInfer)] + " ")

    negativeInferInputIndexes = trainSentIndexNAnsInputRnn[randomNegativeSentIndex][0]

    for _ in range(20):
        negativeInfer, negativeInferInputIndexes = \
            nextInfer(sess, inference, negativeInferInputIndexes, negativeInfer, 0, numMorph)
        sys.stdout.write(morphs[int(negativeInfer)] + " ")

        if morphs[int(negativeInfer)] == "<EOS>":
            break

    sys.stdout.write("\n")