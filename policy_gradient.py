"""

Implement a policy gradient learning to train the agent

"""

import tic_tac_toe as ttt
import numpy as np
from policy_gradient_functions import discount_and_normalize_rewards
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

# Specify the neural network architecture
n_inputs = 9
n_hidden = 18
n_outputs = 9
initializer = tf.contrib.layers.variance_scaling_initializer()
learning_rate = 0.01

# Build the neural network
X = tf.placeholder(tf.float32, shape=[None, n_inputs], name='X')
hidden = fully_connected(X, n_hidden, activation_fn=tf.nn.elu, weights_initializer=initializer)
logits = fully_connected(hidden, n_outputs, activation_fn=None, weights_initializer=initializer)
outputs = tf.nn.softmax(logits)

# Select a random action based on the probability
action = tf.multinomial(tf.log(outputs), num_samples=1)

# Define the target if the action chosen was correct and the cost function
y = np.zeros(n_outputs)
y = tf.reshape(tf.one_hot(depth=n_outputs, indices=action), [1, n_outputs])

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)

# Define gradients
optimizer = tf.train.AdamOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(cross_entropy)
gradients = [grad for grad, variable in grads_and_vars]
gradient_placeholders = []
grads_and_vars_feed = []
for grad, variable in grads_and_vars:
    gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))
training_op = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_iterations = 250
n_games_per_updates = 10  # Number of games before updating the policy
n_max_steps = 100         # Max steps per episode
discount_rate = 0.9       #
save_iterations = 10

env = ttt.make()

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        all_rewards = []
        all_gradients = []
        for game in range(n_games_per_updates):
            current_rewards = []
            current_gradients = []
            obs = env.reset()
            for step in range(n_max_steps):
                action_val, gradients_val = sess.run([action, gradients], feed_dict={X: obs.reshape(1, n_inputs)})
                obs, reward, done, info = env.step(int(action_val[0][0]))
                current_rewards.append(reward)
                current_gradients.append(gradients_val)
                if done:
                    break
            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)

        # Now update the policy
        all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)
        feed_dict = {}
        for var_index, grad_placeholder in enumerate(gradient_placeholders):
            # Multiply the gradients by the action scores and compute the mean
            mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
                                      for game_index, rewards in enumerate(all_rewards)
                                      for step, reward in enumerate(rewards)], axis=0)
            feed_dict[grad_placeholder] = mean_gradients
        sess.run(training_op, feed_dict=feed_dict)
        if iteration % save_iterations == 0:
            saver.save(sess, './policy_net_pg.ckpt')

# env = ttt.make()
# obs = env.reset()
#
# with tf.Session() as sess:
#     saver.restore(sess, './policy_net_pg.ckpt')
#     for step in range(100):
#         action_val = sess.run(action, feed_dict={X: obs.reshape(1, n_inputs)})
#         obs, _, _, _ = env.step(int(action_val[0][0]))
#         env.render()
