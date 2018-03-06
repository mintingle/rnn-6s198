import json
from tqdm import tqdm

import numpy as np
import tensorflow as tf
from nltk import word_tokenize
from sklearn.preprocessing import LabelBinarizer

with open("../data/glove.json", "rt") as fin:
    glove = json.load(fin)
with open("../data/newsgroups.json", "rt") as fin:
    newsgroups = json.load(fin)

def word_vector(text):
    return [glove[token] for token in word_tokenize(text.lower()) if token in glove]
newsgroups["trainX"] = list(map(word_vector, tqdm(newsgroups["trainX"])))
newsgroups["testX"] = list(map(word_vector, tqdm(newsgroups["testX"])))

encoder = LabelBinarizer()
newsgroups["trainY"] = encoder.fit_transform(newsgroups["trainY"])
newsgroups["testY"] = encoder.transform(newsgroups["testY"])

# -----

NHIDDEN = 50                                     # we use a 50d word embedding
NLABELS = len(newsgroups["trainY"][0])           # the number of output classes
MAXTIME = max(map(len, newsgroups["trainX"]))    # the longest sequence so far
print(NHIDDEN, NLABELS, MAXTIME)

x = tf.placeholder(dtype=tf.float32, shape=[1, None, NHIDDEN])
y = tf.placeholder(dtype=tf.float32, shape=[1, NLABELS])

lstm = tf.contrib.rnn.BasicLSTMCell(NHIDDEN)
initial_state = lstm.zero_state(1, tf.float32)
outputs, final_state = tf.nn.dynamic_rnn(cell=lstm, inputs=x, initial_state=initial_state)
logits = tf.contrib.layers.linear(final_state.h, NLABELS)
softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)

predictions = tf.argmax(logits, axis=-1)
loss = tf.reduce_mean(softmax_cross_entropy)

train_op = tf.train.AdamOptimizer().minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

NEPOCH = 2
for step in range(NEPOCH + 1):
    losses = []
    for batchX, batchY in tqdm(zip(newsgroups["trainX"], newsgroups["trainY"])):
        loss_out, _ = sess.run([loss, train_op],
                               feed_dict={
                                   x: np.array([batchX]),
                                   y: batchY[None,:],
                               })
        losses.append(loss_out)
        if len(losses) > 10: break
    print('Loss at step {}: {}'.format(step, sum(losses) / len(losses)))

saver = tf.train.Saver()
path = saver.save(sess, "./tfmodel", global_step=step)
print('Saved checkpoint at {}'.format(path))
print(tf.trainable_variables())
