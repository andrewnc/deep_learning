import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data import get_data
from model import network_model
from tqdm import tqdm

#ssh -N -f -L localhost:8888:localhost:8889 robert@192.168.23.41

def train(train_images, train_labels, test_images, test_labels):
	tf.reset_default_graph()
	sess = tf.Session()

	input_data  = tf.placeholder(tf.float32, [1, 512, 512, 3])
	output_data = tf.placeholder(tf.float32, [1, 512, 512, 1])

	with tf.name_scope( "model" ):
		#so I want the output of this network to be a [1, 512, 512, 1] which we would then pass through the sigmoid which would give us 
		# the probabilities of being cancerous or not (since it's really only a two class problem). We then would use this (maybe by
		# some masking to see if the probability is high enough) along with the label to see if we are correct
	    output = network_model( input_data )

	with tf.name_scope("loss"):
		loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(output, [1,512*512]), labels=tf.reshape(output_data, [1,512*512])))

	optim = tf.train.AdamOptimizer(0.0001).minimize(loss)


	saver = tf.train.Saver()
	sess.run( tf.global_variables_initializer() )

	loss_ans = 0.0

	for i in tqdm(range(1000)):
		ind = np.random.randint(0,20)
		_, loss_val = sess.run([optim,loss], feed_dict={input_data: train_images[ind].reshape([1,512,512,3]), output_data: train_labels[ind].reshape([1,512,512, 1])})

		loss_ans += loss_val
		if i %10 == 0:
			tqdm.write(str(loss_ans/10.0))
			loss_ans = 0.0


if __name__ == "__main__":
	train_images, train_labels, test_images, test_labels = get_data(20,2)
	train(train_images, train_labels, test_images, test_labels)
