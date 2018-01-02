from tensorflow.python.ops.rnn_cell import BasicLSTMCell, GRUCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops.rnn_cell_impl import _linear
from tensorflow.python.ops import array_ops, init_ops, math_ops

from tensorflow.contrib import seq2seq
from tensorflow.contrib import legacy_seq2seq
## other imports
import tensorflow as tf
import numpy as np

from textloader import TextLoader


#
# -------------------------------------------
#
# Global variables

batch_size = 50
sequence_length = 50

data_loader = TextLoader( ".", batch_size, sequence_length )

vocab_size = data_loader.vocab_size  # dimension of one-hot encodings
state_dim = 128

num_layers = 2

tf.reset_default_graph()

#
# ==================================================================
# ==================================================================
# ==================================================================
#

from tensorflow.python.ops.rnn_cell import RNNCell
 




class mygru(RNNCell):
    """GRUCell modeled after the implementation found in rnn_cell for the BasicLSTMCell, it uses _Linear to simplify 
    the process of the linear projects to different dimensions (size of h)"""
 
    def __init__(self, num_units):
        self._num_units = num_units
 
    @property
    def state_size(self):
        return self._num_units
 
    @property
    def output_size(self):
        return self._num_units
 
    def __call__(self, inputs, h, scope=None):
        with tf.variable_scope('GRUCell'):
            with tf.variable_scope('gates'):
                bias = init_ops.constant_initializer(1.0, dtype=tf.float32)
                value = math_ops.sigmoid(_linear([inputs, h],2 * self._num_units,True, bias_initializer=bias))

                #value = math_ops.sigmoid(W([inputs, h]))
                r, z = array_ops.split(value=value, num_or_size_splits=2, axis=1)

            with tf.variable_scope("memory"):
                h_tilde = math_ops.tanh(_linear( [inputs, r*h], self._num_units, True, bias_initializer=bias))
                #h_tilde = math_ops.tanh(W_2([inputs, r*h]))
            h_out = z * h + (1 - z) * h_tilde
            return h_out, h_out

# define placeholders for our inputs.  
# in_ph is assumed to be [batch_size,sequence_length]
# targ_ph is assumed to be [batch_size,sequence_length]

in_ph = tf.placeholder( tf.int32, [ batch_size, sequence_length ], name='inputs' )
targ_ph = tf.placeholder( tf.int32, [ batch_size, sequence_length ], name='targets' )
in_onehot = tf.one_hot( in_ph, vocab_size, name="input_onehot" )

inputs = tf.split( in_onehot, sequence_length, axis=1 )
inputs = [ tf.squeeze(input_, [1]) for input_ in inputs ]
targets = tf.split( targ_ph, sequence_length, axis=1 )

# at this point, inputs is a list of length sequence_length
# each element of inputs is [batch_size,vocab_size]

# targets is a list of length sequence_length
# each element of targets is a 1D vector of length batch_size

# ------------------
# YOUR COMPUTATION GRAPH HERE

with tf.variable_scope("rnn", reuse=None):
    # create a BasicLSTMCell
    #   use it to create a MultiRNNCell
    #   use it to create an initial_state
    #     note that initial_state will be a *list* of tensors!
    # lstm_cell = BasicLSTMCell(state_dim)
    cells = []
    for _ in range(num_layers):
        cell = GRUCell(state_dim)
        cells.append(cell)

    rnn_cells = MultiRNNCell(cells, state_is_tuple=True) # not sure if I have to set the tuple state parameter
    initial_state = rnn_cells.zero_state(batch_size, tf.float32)

    # call seq2seq.rnn_decoder
    outputs, final_state = legacy_seq2seq.rnn_decoder(inputs, initial_state, rnn_cells)
    # transform the list of state outputs to a list of logits.
    # use a linear transformation.
    weights = tf.get_variable('weights', shape=[state_dim, vocab_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
    logits = [tf.matmul(output, weights) for output in outputs]

    # call seq2seq.sequence_loss
    W = tf.convert_to_tensor([tf.ones(shape=[batch_size], dtype=tf.float32)] * len(logits))
    logits = tf.convert_to_tensor(logits)
    targets = tf.convert_to_tensor(targets)
    # print(type(W), type(logits), type(targets))
    targets = tf.reshape(targets, [batch_size, sequence_length])
    loss = seq2seq.sequence_loss(logits, targets, W)

    # create a training op using the Adam optimizer
    optim = tf.train.AdamOptimizer(0.01).minimize(loss)

# ------------------
# YOUR SAMPLER GRAPH HERE

# place your sampler graph here it will look a lot like your
# computation graph, except with a "batch_size" of 1.

# remember, we want to reuse the parameters of the cell and whatever
# parameters you used to transform state outputs to logits!

s_inputs = tf.placeholder(tf.int32, [1], name="s_")
s_input = [tf.one_hot(s_inputs, vocab_size, name="s_onehot")]
with tf.variable_scope("rnn", reuse=True):
    s_initial_state = rnn_cells.zero_state(1, tf.float32)

    s_outputs, s_final_state = legacy_seq2seq.rnn_decoder(s_input, s_initial_state, rnn_cells)

    s_weights = tf.get_variable("weights", shape=[state_dim, vocab_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
    s_probs = tf.nn.softmax(tf.matmul(s_outputs[0], s_weights))

#
# ==================================================================
# ==================================================================
# ==================================================================
#

def sample( num=200, prime='ab' ):

    # prime the pump 

    # generate an initial state. this will be a list of states, one for
    # each layer in the multicell.
    s_state = sess.run( s_initial_state )

    # for each character, feed it into the sampler graph and
    # update the state.
    for char in prime[:-1]:
        x = np.ravel( data_loader.vocab[char] ).astype('int32')
        feed = { s_inputs:x }
        for i, s in enumerate( s_initial_state ):
            feed[s] = s_state[i]
        s_state = sess.run( s_final_state, feed_dict=feed )

    # now we have a primed state vector; we need to start sampling.
    ret = prime
    char = prime[-1]
    for n in range(num):
        x = np.ravel( data_loader.vocab[char] ).astype('int32')

        # plug the most recent character in...
        feed = { s_inputs:x }
        for i, s in enumerate( s_initial_state ):
            feed[s] = s_state[i]
        ops = [s_probs]
        ops.extend( list(s_final_state) )

        retval = sess.run( ops, feed_dict=feed )

        s_probsv = retval[0]
        s_state = retval[1:]

        # ...and get a vector of probabilities out!

        # now sample (or pick the argmax)
        # sample = np.argmax( s_probsv[0] )
        sample = np.random.choice( vocab_size, p=s_probsv[0] )

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
sess.run( tf.global_variables_initializer() )
# summary_writer = tf.summary.FileWriter( "./tf_logs", graph=sess.graph )
saver = tf.train.Saver()
# saver.restore( sess, "./model_final" )

lts = []

print "FOUND %d BATCHES" % data_loader.num_batches

out_text = ""
out_info = ""
for j in range(1000):

    state = sess.run( initial_state )
    data_loader.reset_batch_pointer()

    for i in range( data_loader.num_batches ):
        
        x,y = data_loader.next_batch()

        # we have to feed in the individual states of the MultiRNN cell
        feed = { in_ph: x, targ_ph: y }
        for k, s in enumerate( initial_state ):
            feed[s] = state[k]

        ops = [optim,loss]
        ops.extend( list(final_state) )

        # retval will have at least 3 entries:
        # 0 is None (triggered by the optim op)
        # 1 is the loss
        # 2+ are the new final states of the MultiRNN cell
        retval = sess.run( ops, feed_dict=feed )

        lt = retval[1]
        state = retval[2:]

        if i%1000==0:
            # with open("output_info.txt", "w") as f:
            print "%d %d\t%.4f\n" % ( j, i, lt )    
                # print >> f, out_info
            lts.append( lt )
            saver.save(sess, './model_final')

    # with open("output.txt", "w") as f:
    #     out_text += "{}\n".format(j) +  sample( num=60, prime="I ") + "\n"
    #     print >> f, out_text
    #print sample( num=60, prime="ababab" )
    #print sample( num=60, prime="foo ba" )
    #out_text += sample( num=60, prime="abcdab" )

# summary_writer.close()

#
# ==================================================================
# ==================================================================
# ==================================================================
#

#import matplotlib
#import matplotlib.pyplot as plt
#plt.plot( lts )
#plt.show()
