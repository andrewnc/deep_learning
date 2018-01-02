from textloader import TextLoader
# from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import array_ops
from tensorflow.contrib import seq2seq
from tensorflow.contrib import legacy_seq2seq

from tensorflow.python.ops.rnn_cell import _linear
from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.python.ops.rnn_cell import BasicLSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

import tensorflow as tf
import numpy as np

from string import letters
from random import choice



class GRUCell(RNNCell):
 
    def __init__(self, num_units):
        self._num_units = num_units
 
    @property
    def state_size(self):
        return self._num_units
 
    @property
    def output_size(self):
        return self._num_units
 
    def __call__(self, inputs, h):
        with tf.variable_scope('GRUCell'):
            with tf.variable_scope('gates'):
                temp = _linear([inputs, h], 2 * self._num_units, True, 1.0)
                _r, _z = array_ops.split(1, 2, temp)
                r, z  = sigmoid(_r), sigmoid(_z)
            with tf.variable_scope('memory'):
                temp = _linear([inputs, r * h], self._num_units, True, 1.0)
                h_tilda = tanh(temp)
            new_h = z * h + (1 - z) * h_tilda
        return new_h, new_h


class mygru( RNNCell ):
 
    def __init__( self, state_dim ):
        self.state_dim = state_dim
 
    @property
    def state_size(self):
        return self.state_dim
 
    @property
    def output_size(self):
        return self.state_dim

 
    def __call__( self, inputs, h, scope=None ):
        w_xr = tf.get_variable("xr_weights", [inputs.shape[1], h.shape[1]], initializer=tf.truncated_normal_initializer(stddev=0.1))
        w_hr = tf.get_variable("hr_weights", [h.shape[1], h.shape[1]], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_r = tf.get_variable("r_bias", [1, h.shape[1]], initializer=tf.ones_initializer)
        r = math_ops.sigmoid(tf.matmul(inputs, w_xr) + tf.matmul(h, w_hr) + b_r)

        w_xz = tf.get_variable("xz_weights", [inputs.shape[1], h.shape[1]], initializer=tf.truncated_normal_initializer(stddev=0.1))
        w_hz = tf.get_variable("hz_weights", [h.shape[1], h.shape[1]], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_z = tf.get_variable("z_bias", [h.shape[0], h.shape[1]], initializer=tf.ones_initializer)
        z = math_ops.sigmoid(tf.matmul(inputs, w_xz) + tf.matmul(h, w_hz) + b_z)

        w_xh = tf.get_variable("xh_weights", [inputs.shape[1], h.shape[1]], initializer=tf.truncated_normal_initializer(stddev=0.1))
        w_hh = tf.get_variable("hh_weights", [h.shape[1], h.shape[1]], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_h  = tf.get_variable("h_bias", [1, h.shape[1]], initializer=tf.ones_initializer)
        h_tilde = math_ops.tanh(tf.matmul(inputs, w_xh) + tf.matmul(r*h, w_hh) + b_h)

        h_out = z * h + (1-z)*h_tilde
        return h_out, h_out #return double
        
#
# -------------------------------------------
#
# Global variables

batch_size = 100
sequence_length = 100

data_loader = TextLoader(".", batch_size, sequence_length)

vocab_size = data_loader.vocab_size  # dimension of one-hot encodings
state_dim = 128

num_layers = 2

#
# ==================================================================
# ==================================================================
# ==================================================================
#

# define placeholders for our inputs.  
# in_ph is assumed to be [batch_size,sequence_length]
# targ_ph is assumed to be [batch_size,sequence_length]

in_ph = tf.placeholder(tf.int32, [batch_size, sequence_length], name='inputs')
targ_ph = tf.placeholder(tf.int32, [batch_size, sequence_length], name='targets')
in_onehot = tf.one_hot(in_ph, vocab_size, name="input_onehot")

inputs = tf.split( in_onehot, sequence_length, axis=1 )
inputs = [ tf.squeeze(input_, [1]) for input_ in inputs ]
targets = tf.split( targ_ph, sequence_length, axis=1 )

# at this point, inputs is a list of length sequence_length
# each element of inputs is [batch_size,vocab_size]

# targets is a list of length sequence_length
# each element of targets is a 1D vector of length batch_size

# ------------------
# COMPUTATION GRAPH

with tf.variable_scope('rnn', reuse=None):
    # create a BasicLSTMCell
    #   use it to create a MultiRNNCell
    #   use it to create an initial_state
    #     initial_state will be a *list* of tensors
    gru_cell = GRUCell(state_dim)
    rnn_cells = MultiRNNCell([gru_cell] * num_layers, state_is_tuple=True)
    initial_state = rnn_cells.zero_state(batch_size, tf.float32)

    # call seq2seq.rnn_decoder
    outputs, final_state = legacy_seq2seq.rnn_decoder(inputs, initial_state, rnn_cells)

    # transform the list of state outputs to a list of logits.
    # use a linear transformation.
    trainable_weights = tf.get_variable('trainable_w', shape=[state_dim, vocab_size],
        initializer=tf.truncated_normal_initializer(stddev=0.1))
    logits = [tf.matmul(output, trainable_weights) for output in outputs]

    # call seq2seq.sequence_loss
    weights = [tf.ones(shape=[batch_size], dtype=tf.float32)] * len(logits)
    loss = seq2seq.sequence_loss(logits, targets, weights)

    # create a training op using the Adam optimizer
    optim = tf.train.AdamOptimizer(0.01).minimize(loss)


# ------------------
# SAMPLER GRAPH

# place your sampler graph here it will look a lot like your
# computation graph, except with a "batch_size" of 1.

# remember, we want to reuse the parameters of the cell and whatever
# parameters you used to transform state outputs to logits!

s_inputs = tf.placeholder(tf.int32, [1], name='x_')
s_input = [tf.one_hot(s_inputs, vocab_size, name="x_onehot")]

with tf.variable_scope('rnn', reuse=True):
    s_initial_state = rnn_cells.zero_state(1, tf.float32)
    # call seq2seq.rnn_decoder
    s_outputs, s_final_state = legacy_seq2seq.rnn_decoder(s_input, s_initial_state, rnn_cells)

    s_weights = tf.get_variable('trainable_w', shape=[state_dim, vocab_size],
        initializer=tf.random_normal_initializer(stddev=0.1))
    s_probs = tf.nn.softmax(tf.matmul(s_outputs[0], s_weights))

#
# ==================================================================
# ==================================================================
# ==================================================================
#

def sample(num=200, prime='ab'):

    # prime the pump 

    # generate an initial state. this will be a list of states, one for
    # each layer in the multicell.
    s_state = sess.run(s_initial_state)

    # for each character, feed it into the sampler graph and
    # update the state.
    for char in prime[:-1]:
        x = np.ravel(data_loader.vocab[char]).astype('int32')
        feed = {s_inputs:x}
        for i, s in enumerate(s_initial_state):
            feed[s] = s_state[i]
        s_state = sess.run(s_final_state, feed_dict=feed)

    # now we have a primed state vector; we need to start sampling.
    ret = prime
    char = prime[-1]
    for n in range(num):
        x = np.ravel(data_loader.vocab[char]).astype('int32')

        # plug the most recent character in...
        feed = {s_inputs:x}
        for i, s in enumerate(s_initial_state):
            feed[s] = s_state[i]
        ops = [s_probs]
        ops.extend(list(s_final_state))

        retval = sess.run(ops, feed_dict=feed)
        s_probsv = retval[0]
        s_state = retval[1:]

        # ...and get a vector of probabilities out!

        # now sample (or pick the argmax)
        # sample = np.argmax( s_probsv[0] )
        sample = np.random.choice(vocab_size, p=s_probsv[0])

        pred = data_loader.chars[sample]
        ret += pred
        char = pred

    return ret

#
# ==================================================================
# ==================================================================
# ==================================================================
#

sess = tf.Session()
sess.run(tf.initialize_all_variables())
summary_writer = tf.train.SummaryWriter("./tf_logs", graph=sess.graph)

lts = []

print "FOUND %d BATCHES" % data_loader.num_batches

for j in range(1000):

    state = sess.run(initial_state)
    data_loader.reset_batch_pointer()

    for i in range(data_loader.num_batches):
        
        x,y = data_loader.next_batch()
        # we have to feed in the individual states of the MultiRNN cell
        feed = {in_ph: x, targ_ph: y}
        for k, st in enumerate(initial_state):
            feed[st] = state[k]
        ops = [optim, loss]
        ops.extend(list(final_state))

        # retval will have at least 3 entries:
        # 0 is None (triggered by the optim op)
        # 1 is the loss
        # 2+ are the new final states of the MultiRNN cell
        _retval = sess.run(ops, feed_dict=feed)

        lt = _retval[1]
        state = _retval[2:]

        if i%1000 == 0:
            print "%d %d\t%.4f" % (j, i, lt)
            lts.append(lt)

    print sample(num=140, prime=choice(letters).upper())
    print sample(num=140, prime='@')
#    print sample(num=60, prime="And ")
#    print sample( num=60, prime="ababab" )
#    print sample( num=60, prime="foo ba" )
#    print sample( num=60, prime="abcdab" )

summary_writer.close()

#
# ==================================================================
# ==================================================================
# ==================================================================
#

#import matplotlib
#import matplotlib.pyplot as plt
#plt.plot( lts )
#plt.show()