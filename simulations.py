"""

Simulations to evaluate model quality

"""

import tic_tac_toe as ttt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

# Specify the neural network architecture
n_inputs = 9
n_hidden = 18
n_outputs = 9
initializer = tf.contrib.layers.variance_scaling_initializer()
learning_rate = 0.001

# Build the neural network
X = tf.placeholder(tf.float32, shape=[None, n_inputs], name='X')
hidden = fully_connected(X, n_hidden, activation_fn=tf.nn.elu, weights_initializer=initializer)
hidden2 = fully_connected(hidden, n_hidden, activation_fn=tf.nn.elu, weights_initializer=initializer)
logits = fully_connected(hidden2, n_outputs, activation_fn=None, weights_initializer=initializer)

possible_moves_mask = tf.ones(shape=[1, n_outputs], dtype=tf.float32) - tf.abs(X)
temp = tf.multiply(possible_moves_mask, tf.exp(logits))
outputs = 1/tf.reduce_sum(temp)*temp

# Select a random action based on the probability
action = tf.multinomial(tf.log(outputs), num_samples=1)

# Define the target if the action chosen was correct and the cost function
y = tf.reshape(tf.one_hot(depth=n_outputs, indices=action), [1, n_outputs])
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)

# Define gradients
optimizer = tf.train.AdamOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(cross_entropy)
gradients = [grad for grad, variable in grads_and_vars]
variables_names = [v.name for v in tf.trainable_variables()]

gradient_placeholders = []
grads_and_vars_feed = []
for grad, variable in grads_and_vars:
    gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))
training_op = optimizer.apply_gradients(grads_and_vars_feed)

# End of the construction phase
names = [n.name for n in tf.get_default_graph().as_graph_def().node]
print(names)
init = tf.global_variables_initializer()
# To do : we don't need to save all variables
saver = tf.train.Saver()

env = ttt.make_simulation(begin=True)
obs = env.reset()

win, tie, lose = 0, 0, 0

with tf.Session() as sess:
    saver.restore(sess, './policy_net_pg_improved.ckpt')
    for game in range(100):
        obs = env.reset()
        for step in range(100):
            obs = obs.reshape(1, n_inputs)
            mask = 1 - np.abs(obs)
            action_val = sess.run(action, feed_dict={X: obs, possible_moves_mask: mask})
            obs, reward, done, info = env.step(int(action_val[0][0]))
            if done:
                if reward == 5:
                    win += 1
                elif reward == -5:
                    lose += 1
                else:
                    tie += 1
                break
print(win, tie, lose)