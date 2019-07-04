import sys
import random
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

from Week03_Seq2Seq import train_morph_file_path, \
    w2v_gensim_model_file_path

from Week03_Seq2Seq import trainSentIndexNAnsInputRnn, morphs, numMorph, seqLen

from Week03_Seq2Seq import indexesToInputOneHots, batchIndexesToInputOneHots
from Week03_Seq2Seq import indexesToOutputOneHots, batchIndexesToOutputOneHots

def inferOne(tf_sess, tf_inference, morphIndexes, ans, numMorph):
    indexesNAnsList = [(morphIndexes, ans)]
    inputForTF = batchIndexesToInputOneHots(indexesNAnsList, numMorph)
    infer = tf_sess.run(tf_inference, feed_dict={x: inputForTF})

    return infer


def pickRandomSentIndexToStart(ans):


def inferSeq(tf_sess, tf_inference, ans, numMorph):
    randomSentIndex = -1
    while randomSentIndex < 0:
        randomSentIndex = random.randrange(0, len(trainSentIndexNAnsInputRnn))
        if trainSentIndexNAnsInputRnn[randomSentIndex][0][0] != 0: # <BOS-8>
            randomSentIndex = -1
        if trainSentIndexNAnsInputRnn[randomSentIndex][1] != ans:
            randomSentIndex = -1

    randomSentIndex = pickRandomSentIndexToStart(ans)
    totalSeq = trainSentIndexNAnsInputRnn[randomSentIndex][0]

    for _ in range(20):
        if morphs[int(totalSeq[-1])] == "<EOS>":
            break

        nextInfer = inferOne(tf_sess, tf_inference, totalSeq[-seqLen:], ans, numMorph)
        totalSeq.append(nextInfer)

    if ans == 1:
        sys.stdout.write("Positive - ")
    else:
        sys.stdout.write("Negative - ")

    for idx in totalSeq[seqLen-1:]:
        sys.stdout.write(morphs[int(idx)] + " ")

    sys.stdout.write("\n")


def define_argparser():
    p = ArgumentParser()
    p.add_argument('-e', type=int, default=10, help="the epoch number of model to restore")

    config = p.parse_args()

    return config



if __name__ == "__main__":
    args = define_argparser()

    sess = tf.Session()

    saver = tf.train.import_meta_graph('./model/rnn_gen_%03d.meta' % args.e)
    saver.restore(sess, './model/rnn_gen_%03d' % args.e)


    graph = tf.get_default_graph()

    x = graph.get_tensor_by_name("x:0")
    inference = graph.get_tensor_by_name("inference:0")

    for _ in range(5):
        inferSeq(sess, inference, 1, numMorph) # positive

    for _ in range(5):
        inferSeq(sess, inference, 0, numMorph) # negative