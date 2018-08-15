import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from tensorflow.python import debug as tf_debug

# Returns uniformly generated numpy array with generate [ x | t | random] where x and t are shuffled
# Shape of the generated x_arr and generated t_arr should be the same
def generateData(x_start, x_end, dx, t_start, t_end, dt):
	x_arr = np.arange(start=x_start, stop=x_end, step=dx, dtype=np.float32).reshape(-1,1)
	np.random.shuffle(x_arr)
	t_arr = np.arange(start=t_start, stop=t_end, step=dt, dtype=np.float32).reshape(-1,1)
	np.random.shuffle(t_arr)
	rand = np.random.random_sample(x_arr.shape)
	return np.concatenate((x_arr, t_arr, rand), axis=1)

#Extract specified portion of data set for training or validation from [ x | t | random]
#train_set=False extracts the validation data
#Threshold to select portion of training data -> if threshold = .8, Training data are rows with randoms with < .8 and validation data is >= .8
def extract(arr, threshold, train_set=True):
	if train_set:
		condition = arr[:, 2:] < threshold
	else:
		condition = arr[:, 2:] >= threshold

	#extract x and t where the random column fits the condition
	x = np.extract(condition, arr[:, :1]).reshape(-1,1)	
	t = np.extract(condition, arr[:, 1:2]).reshape(-1,1)
	return np.concatenate((x, t), axis=1)

## Set up psi and PDE function in terms of tensorflow - The NN output (out_layer) and Psi function will be defined later on ##
x = tf.placeholder(shape=[None,1],dtype=tf.float32)
t = tf.placeholder(shape=[None,1],dtype=tf.float32)
one = tf.constant(1.0, dtype=tf.float32)
two = tf.constant(2.0, dtype=tf.float32)
five = tf.constant(5.0, dtype=tf.float32)
T = tf.constant(10.0, dtype=tf.float32)
volatility = tf.constant(.2, dtype=tf.float32)
half = tf.constant(.5, dtype=tf.float32)
r = tf.constant(.01, dtype=tf.float32)

A1 = (((two - x) / two) * (tf.exp(-r*(T-t)) * (one / five)))	
A2 = ((x / two) * (tf.exp(-r*(T-t)) * (tf.exp(two) / five))) 
A3 = ((T-t)*((tf.exp(x)/five) - (((two - x) / two) * (one / five) + (x / two) * (tf.exp(two) / five))))
A = A1 + A2 + A3

## Network parameters ##
hidden_1_size = 10	#neurons in hidden layer 1
input_size = 2
output_size = 1 

## Learning parameters ##
learn_rate = 1.2
total_batch_size = 500 #includes both training set & validation set
training_epochs = 100

## Generate input and output data ##
dx = float(2) / total_batch_size
dt = float(10) / total_batch_size

data_set = generateData(0, 2, dx, 0, 10, dt) # [ x | t | random]
in_batch = extract(data_set, .8)
x_batch = in_batch[:, :1]
t_batch = in_batch[:, 1:]

batch_size = in_batch.shape[0]	#training batch size
output_batch = np.zeros(shape=(batch_size, output_size), dtype=np.float32)

## Tensorflow Graph ##
n_input = tf.placeholder(name="n_input", shape=[None, input_size], dtype=tf.float32)

weights = {
	'w1' : tf.Variable(tf.random_normal([input_size, hidden_1_size]), name='w1'),
	'w_out': tf.Variable(tf.random_normal([hidden_1_size, output_size]), name='w_out')
}

biases = {
	'b1' : tf.Variable(tf.random_normal([hidden_1_size]), name='b1'),
	'b_out': tf.Variable(tf.random_normal([output_size]), name='b_out')
}

layer_1 = tf.sigmoid(tf.matmul(n_input, weights['w1']) + biases['b1'])
out_layer = tf.sigmoid(tf.matmul(layer_1, weights['w_out']) + biases['b_out'], name="nn")


# Psi
psi = A + (x * (2 - x) * (T - t) * out_layer)


## Loss ##
# Calculate the Hessian Matrix and convert to a usable form
h = tf.hessians(ys=psi, xs=[x])
x_hess = tf.reshape(h[0], (batch_size, batch_size))
x_hess_diag = tf.reshape(tf.diag_part(x_hess), (batch_size, 1))

dpsi_dt = tf.gradients(psi, [t], stop_gradients=[x]) 
dpsi_dx = tf.gradients(psi, [x], stop_gradients=[t])

loss = dpsi_dt + (half * tf.square(volatility) * tf.square(x) * x_hess_diag) + (r * x * dpsi_dx) - (r * psi)
cost = tf.reduce_mean(tf.square(loss)) #taking mean of loss tensor seems to work best
#cost = tf.reduce_sum(tf.square(loss))
#cost = tf.square(loss)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(cost)

#Launch graph
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer()) #initializes weights, biases

	# Train NN
	for epoch in range(training_epochs):
		_, l = sess.run([optimizer, cost], feed_dict={ x: x_batch, t: t_batch, n_input: in_batch})
		print("Epoch %s loss: " % epoch, l)
		''' #Prints weights and bias per training epoch
		w=sess.run(weights)
		print("Weights: \n",w)
		b=sess.run(biases)
		print("Bias: \n",b)
		'''

	# Validation using remaining data
	test_batch = extract(data_set, .8, train_set=False)
	x_test = test_batch[:, :1]
	t_test = test_batch[:, 1:]

	N = sess.run(out_layer, feed_dict={n_input: test_batch})
	prediction = A + (x * (2 - x) * (T - t) * N)
	results = sess.run(prediction, feed_dict={x: x_test, t: t_test})
	print("Validation")
	print("X | T | Results:\n", np.concatenate((test_batch, results), axis=1))

	#Save Model - creates 4 files in same directory
	saver = tf.train.Saver()
	saver.save(sess, './model')


## Plot and save plots ##
# Convert data to 1D for plotting
X = x_test.flatten()
T = t_test.flatten()
R = results.flatten()

# X vs R and T vs R
plt.subplot(121)
plt.gca().set_title("X vs R")
plt.scatter(X, R)
plt.xlabel("X")
plt.ylabel("output")

plt.subplot(122)
plt.gca().set_title("T vs R")
plt.scatter(T, R)
plt.xlabel("T")
plt.ylabel("output")
plt.savefig("validation.png")

fig2 = plt.figure()
fig3 = plt.figure()

# Trisurface plot
ax = fig2.add_subplot(111,projection='3d')
ax.plot_trisurf(T, X, R, cmap='viridis', edgecolor='none')
ax.set_title('Trisurface Plot')
ax.set_xlabel('T')
ax.set_ylabel('X')
ax.set_zlabel('Option Value')
fig2.savefig('v_trisurf.png')

#Scatter plot
ax2 = fig3.add_subplot(111, projection='3d')
ax2.scatter(T, X, R, c=R, cmap='viridis', linewidth=0.5)
ax2.set_title('Scatter Plot')
ax2.set_xlabel('T')
ax2.set_ylabel('X')
ax2.set_zlabel('Option Price')
fig3.savefig('v_scatter.png')

plt.show()