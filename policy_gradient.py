"""

Implement a policy gradient learning to train the agent

"""

import tic_tac_toe as ttt
import numpy as np
from policy_gradient_functions import discount_and_normalize_rewards
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from datetime import datetime

# Path and date variables setup for tensorboard logs
now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_logdir = 'tf_logs'
logdir = '{}/run-{}/'.format(root_logdir, now)

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
names = [n.name for n in tf.get_default_graph().as_graph_def().node]  # Names of all nodes
init = tf.global_variables_initializer()
# Initialize a saver to restore a session
saver = tf.train.Saver()
# Initialize a summary and a file writer for tensorboard visualization
cross_entropy_summary = tf.summary.scalar(name='Cross Entropy Value', tensor=tf.reduce_sum(cross_entropy))
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

n_iterations = 250
n_games_per_updates = 100  # Number of games before updating the policy
n_max_steps = 100          # Max steps per episode
discount_rate = 0.95       #
save_iterations = 10

env = ttt.make_simulation()

with tf.Session() as sess:
    init.run()
    # U
    # saver.restore(sess, './policy_net_pg.ckpt')
    for iteration in range(n_iterations):
        all_rewards = []
        all_gradients = []
        env = ttt.make_simulation(opponent='random', begin=bool(iteration % 2))
        for game in range(n_games_per_updates):
            current_rewards = []
            current_gradients = []
            obs = env.reset()
            for step in range(n_max_steps):
                obs = obs.reshape(1, n_inputs)
                action_val, gradients_val = sess.run([action, gradients], feed_dict={X: obs})
                obs, reward, done, info = env.step(int(action_val))
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
            # Write a summary data point
            summary_str = cross_entropy_summary.eval(feed_dict={X: np.array([1, -1, 0, -1, 1, 0, 0, 0, 0]).reshape(1,9)})
            file_writer.add_summary(summary_str, iteration)
            # Print current iteration
            print(iteration)
            # Save a session checkpoint. Warning : erasing the previous one
            saver.save(sess, './tf_checkpoints/policy_net_pg_improved.ckpt')

    # Close the file writer
    file_writer.close()