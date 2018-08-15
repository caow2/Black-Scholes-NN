import tensorflow as tf
import numpy as np

#Given parameters, evaluate trial solution psi
def psi(x, t, N):
	r = .01
	T = 10
	A = (((2-x)/2) * (np.exp(-r*(T-t)) * (1/5)) +
		((x/2) * (np.exp(-r*(T-t)) * (np.exp(2)/5))) +
		((T-t) * ((np.exp(x)/5) - (((2-x)/2) * (1/5) + (x/2) * (np.exp(2)/5)))))
	psi_x_t = A + (x * (2 - x) * (T - t) * N)
	return psi_x_t



with tf.Session() as sess:
	#Load saved session
	saver = tf.train.import_meta_graph('model.meta')
	saver.restore(sess, tf.train.latest_checkpoint('./'))
	
	#Restore weights and biases
	graph = tf.get_default_graph()
	w1 = graph.get_tensor_by_name('w1:0')
	print('w1:', sess.run(w1))
	w_out = graph.get_tensor_by_name('w_out:0')
	print('w_out', sess.run(w_out))

	b1 = graph.get_tensor_by_name('b1:0')
	print('b1:', sess.run(b1))
	b_out = graph.get_tensor_by_name('b_out:0')
	print('b_out', sess.run(b_out))

	#Restore NN structure
	nn = graph.get_tensor_by_name('nn:0')
	n_input = graph.get_tensor_by_name('n_input:0')
	
	#Ask for input to test NN
	try:
		test = True
		while(test):
			x_in = float(input("Input x: "))
			t_in = float(input("Input t: "))
			n_in = np.array([[x_in, t_in]], dtype=np.float32)
			n_out = sess.run(nn, feed_dict={ n_input: n_in})

			print("N: ", n_out)
			print(psi(x_in, t_in, n_out))
			testing = input("Input another set of values? To stop, enter anything but Y or y \n")
			if testing.upper() != 'Y':
				test = False
	except Exception as ex:
			print(ex)