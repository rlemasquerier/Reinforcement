"""

Main file to play a Tic-Tac-Toe game against an opponent
Current available opponents :
   - random
   - policy_gradient

"""
import tic_tac_toe as ttt

env = ttt.make_simulation(opponent='policy_gradient', begin=False)
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


