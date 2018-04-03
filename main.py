"""

"""
import tic_tac_toe as ttt

env = ttt.make_simulation(opponent='model', begin=False)
obs = env.reset()

rewards = []
for step in range(9):
    env.render()
    print('Where do you play ?')
    player_input = int(input())
    obs, reward, done, info = env.step(player_input)
    rewards.append(reward)
    if done:
        break

print(rewards)


