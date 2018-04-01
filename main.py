"""

"""
import tic_tac_toe as ttt

env = ttt.make()
env.reset()

for play in range(9):
    env.step(env.action_space.sample())
    if env.done:
        break

print(env.render())
