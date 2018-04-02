"""

"""
import tic_tac_toe as ttt
import tensorflow as tf
env = ttt.make()
env.reset()

env.step(3)
env.step(5)

print(env.render())

env = ttt.make()
obs = env.reset()
#
with tf.Session() as sess:
    saver.restore(sess, './policy_net_pg.ckpt')
    for step in range(100):
        action_val = sess.run(action, feed_dict={X: obs.reshape(1, n_inputs)})
        obs, _, _, _ = env.step(int(action_val[0][0]))
        env.render()