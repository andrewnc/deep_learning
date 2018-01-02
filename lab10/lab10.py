from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
import vgg16

from scipy.misc import imread, imresize, imsave

# persistent values
layers = [ 'conv1_1', 'conv1_2',
           'conv2_1', 'conv2_2',
           'conv3_1', 'conv3_2', 'conv3_3',
           'conv4_1', 'conv4_2', 'conv4_3',
           'conv5_1', 'conv5_2', 'conv5_3' ]
content_layer = 'conv4_2'
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']


def get_vgg(sess):
	opt_img = tf.Variable(tf.truncated_normal([1,224,224,3], dtype=tf.float32, 
	                                           stddev=1e-1), name='opt_img')
	tmp_img = tf.clip_by_value(opt_img, 0.0, 255.0)
	vgg = vgg16.vgg16(tmp_img, 'vgg16_weights.npz', sess)
	return vgg, opt_img

def get_style_and_content(sess, vgg, content='content.png', style='style.png'):
	global content_layer
	global style_layers
	global layers
	# load the style image
	style_img = imread(style, mode='RGB')
	style_img = imresize(style_img, (224, 224))
	style_img = np.reshape(style_img, [1,224,224,3])
	# load the content image
	content_img = imread(content, mode='RGB')
	content_img = imresize(content_img, (224, 224))
	content_img = np.reshape(content_img, [1,224,224,3])
	# get content and style activations
	ops = [getattr(vgg, x) for x in layers]
	temp_content = sess.run(ops, feed_dict={vgg.imgs: content_img})
	temp_style = sess.run(ops, feed_dict={vgg.imgs: style_img})
	# extract only the necessary activations
	assert isinstance(content_layer, str)
	content_acts = temp_content[layers.index(content_layer)]
	style_acts = map(lambda _l: temp_style[layers.index(_l)], style_layers)
	return style_acts, content_acts, content_img

def get_grams(style_acts):
	style_grams = []
	for activations in style_acts:
		# compute gram matrices for style layers
		depth = activations.shape[-1]
		temp_layer_acts = activations.reshape(-1, depth)
		gram = np.dot(temp_layer_acts.T, temp_layer_acts)
		style_grams.append(gram)
	return style_grams

def get_style_loss(vgg, style_grams):
	global style_layers
	with tf.name_scope('style'):
		# get style features op from vgg activations
		style_ops = [getattr(vgg, x) for x in style_layers]
		w_l = tf.constant(1/5.0, dtype=tf.float32, name='factor')
		style_losses = []
		for i in xrange(len(style_ops)):
			# get dimentions
			_, w, h, d = map(lambda x: x.value, style_ops[i].get_shape())
			N = d
			M = w * h
			# compute gram matrix for generated image for a layer
			_reshaped_layer = tf.reshape(style_ops[i], [-1, N])
			g_l = tf.matmul(_reshaped_layer, _reshaped_layer, transpose_a=True)
			# compute style loss
			e_l = tf.nn.l2_loss(tf.subtract(g_l, style_grams[i]))
			e_l = tf.truediv(e_l, 2.0*(N**2)*(M**2), name='e_l')
			style_losses.append(tf.multiply(w_l, e_l, name='we_l'))
		# sum all the style losses together
		style_loss = reduce(tf.add, style_losses)
	return style_loss

def get_content_loss(vgg, p_content):
	global content_layer
	with tf.name_scope('content'):
		g_content = getattr(vgg, content_layer)
		content_loss = tf.nn.l2_loss(tf.subtract(g_content, p_content))
	return content_loss


def save_image(path, img):
	img = np.clip(img, 0.0, 255.0).astype(np.uint8)
	imsave(path, img)

# Relevant snippets from the paper:
#   For the images shown in Fig 2 we matched the content representation on layer 'conv4_2'
#   and the style representations on layers 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1' and 'conv5_1'
#   The ratio alpha/beta was  1x10-3
#   The factor w_l was always equal to one divided by the number of active layers (ie, 1/5)

def run():
    # initialzie constants
    content_weight = tf.constant(1, dtype=tf.float32)
    style_weight = tf.constant(10000, dtype=tf.float32)

    sess = tf.Session()
    vgg, opt_img = get_vgg(sess)
    style_acts, content_acts, content_img = get_style_and_content(sess, vgg, 
                                            content='content.png', style='style.png')
    original_style_grams = get_grams(style_acts)
    style_loss = get_style_loss(vgg, original_style_grams)
    content_loss = get_content_loss(vgg, content_acts)


    with tf.name_scope('loss'):
        total_loss = tf.add(content_weight * content_loss, style_weight * style_loss)

    train_step = tf.contrib.opt.ScipyOptimizerInterface(total_loss, var_list=[opt_img], method='L-BFGS-B', options={'maxiter': 0})
    
    # this clobbers all VGG variables, but we need it to initialize the
    # Adam stuff, so we reload all of the weights...
    sess.run(tf.global_variables_initializer())
    vgg.load_weights('vgg16_weights.npz', sess)

    # initialize with the content image
    sess.run(opt_img.assign(content_img))

    # optimization loop
    for step in xrange(int(10e4)):
        if step % 10 == 0:
            l, c, s = sess.run([total_loss, content_loss, style_loss])
            print('i {} total loss {} content loss {} style loss {}'.format(step, l, c, s))
        # take an optimizer step
        train_step.minimize(sess)
        # clip values
        if step % 100 == 0:
            temp_img = sess.run(opt_img)
            save_image('output/{}_{}.png'.format("starry_night", step), temp_img[0])
            temp_img = tf.clip_by_value(temp_img, 0.0, 255.0)
            sess.run(opt_img.assign(temp_img))


    sess.close()

if __name__ == '__main__':
    run()