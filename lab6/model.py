import tensorflow as tf
import numpy as np

def conv( x, filter_size=3, stride=2, num_filters=64, is_output=False, name="conv" ):
    filter_height, filter_width = filter_size, filter_size
    in_channels = x.get_shape().as_list()[-1]
    out_channels = num_filters
    
    with tf.variable_scope(name) as scope:
        W = tf.get_variable("{}_W".format(name), shape=[filter_height, filter_width, in_channels, out_channels],
                            initializer = tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable("{}_b".format(name), shape=[out_channels],
                            initializer = tf.contrib.layers.variance_scaling_initializer())
        conv = tf.nn.conv2d(x, W, [1, stride, stride, 1], padding="SAME")
        out = tf.nn.bias_add(conv, b)
        
        if not is_output:
            out = tf.nn.relu(out)
            
    return out

def convt( x, out_shape, filter_size=8, stride=2, is_output=False, name="convt" ):
    filter_height, filter_width = filter_size, filter_size
    in_channels = x.get_shape().as_list()[-1]

    with tf.variable_scope( name ):
        W = tf.get_variable( "{}_W".format(name), shape=[filter_height, filter_width, out_shape[-1], in_channels],
                            initializer = tf.contrib.layers.variance_scaling_initializer() )
        b = tf.get_variable( "{}_b".format(name), shape=[out_shape[-1]],
                            initializer = tf.contrib.layers.variance_scaling_initializer() )
        conv = tf.nn.conv2d_transpose( x, W, out_shape, [1, stride, stride, 1], padding="SAME" )
        out = tf.nn.bias_add( conv, b )
        if not is_output:
            out =  tf.nn.relu(out)

    return out

def fc( x, out_size=50, is_output=False, name="fc" ):
    in_size = x.get_shape().as_list()[0]
    
    with tf.variable_scope(name) as scope:
        W = tf.get_variable("{}_W".format(name), shape=[out_size, in_size], initializer = tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable("{}_b".format(name), shape=[out_size, 1], initializer = tf.contrib.layers.variance_scaling_initializer())
        
        out = tf.matmul(W, x)
        out = out + b
        
        if not is_output:
            out = tf.nn.relu(out)
    return out

def network_model(input_data):
	# input_data = [1, 512, 512, 3]
	h0 = conv(input_data, stride=1, name="h0") #[1, 512, 512, 64]
	h1 = conv(h0, stride=1, name="h1") #[1, 512, 512, 64]
	output = conv(h1, stride=1, num_filters=1, is_output=True, name="output") #[1, 512, 512, 1]
	return output
