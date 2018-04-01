"""

Implement a policy gradient learning to train the agent

"""

import tic_tac_toe as ttt
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

# Specify the neural network architecture
n_inputs = 9
n_hidden = 18
n_outputs = 9
initializer = tf.contrib.layers.variance_scaling_initializer()

# Build the neural network
X = tf.placeholder(tf.float32, shape=[None, n_inputs], name='X')
hidden = fully_connected(X, n_hidden, activation_fn=tf.nn.elu, weights_initializer=initializer)
logits = fully_connected(hidden, n_outputs, activation_fn=None, weights_initializer=initializer)
outputs = tf.nn.softmax(logits)

# Select a random action based on the probability
action = tf.multinomial(tf.log(outputs), num_samples=1)

init = tf.global_variables_initializer()

env = ttt.make()
env.reset()

with tf.Session() as sess:
    init.run()

    obs = env.observation()
    print(outputs.eval(feed_dict={X: obs.reshape(1, n_inputs)}))





