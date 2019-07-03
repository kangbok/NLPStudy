import sys
import random

import tensorflow as tf

from Week03_Seq2Seq import train_morph_file_path, \
    w2v_gensim_model_file_path

from Week03_Seq2Seq import trainSentIndexNAnsInputRnn, morphs, numMorph

from Week03_Seq2Seq import indexesToInputOneHots, batchIndexesToInputOneHots
from Week03_Seq2Seq import indexesToOutputOneHots, batchIndexesToOutputOneHots


if __name__ == "__main__":
    sess = tf.Session()

    saver = tf.train.import_meta_graph('./model/rnn_gen_012.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./model/'))


    graph = tf.get_default_graph()

    x = graph.get_tensor_by_name("x")
    inference = graph.get_tensor_by_name("inference")

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

    positiveInferInputIndexList = [trainSentIndexNAnsInputRnn[randomPositiveSentIndex]]
    negativeInferInputIndexList = [trainSentIndexNAnsInputRnn[randomNegativeSentIndex]]

    positiveOneX = batchIndexesToInputOneHots(positiveInferInputIndexList, numMorph)
    negativeOneX = batchIndexesToInputOneHots(negativeInferInputIndexList, numMorph)

    positiveInfer = sess.run(inference, feed_dict={x: positiveOneX})
    # positiveInfer = tf.cast(positiveInfer[0], "int")
    negativeInfer = sess.run(inference, feed_dict={x: negativeOneX})
    # negativeInfer = tf.cast(negativeInfer[0], "int")

    sys.stdout.write("Positive - ")
    for idx in trainSentIndexNAnsInputRnn[randomPositiveSentIndex][0]:
        sys.stdout.write(morphs[idx] + " ")
    sys.stdout.write(morphs[int(positiveInfer)] + "\n")

    sys.stdout.write("Negative - ")
    for idx in trainSentIndexNAnsInputRnn[randomNegativeSentIndex][0]:
        sys.stdout.write(morphs[idx] + " ")
    sys.stdout.write(morphs[int(negativeInfer)] + "\n")