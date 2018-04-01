"""

"""

import numpy as np
import random


class ActionSpace:
    def __init__(self, matrix):
        self.space = set()
        for i in range(3):
            for j in range(3):
                if matrix[i, j] == 0:
                    self.space.add((i, j))

    def sample(self):
        return random.sample(self.space, 1)[0]


class Environment:
    def __init__(self):
        self.matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.action_space = ActionSpace(self.matrix)
        self.current_player = 1
        self.done = False

    def reset(self):
        self.matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.action_space = ActionSpace(self.matrix)
        self.current_player = 1

    def render(self):
        print(self.matrix)

    def check_done(self):

        if 0 not in self.matrix:
            self.done = True
            return True

        if self.matrix[0, 0] == self.matrix[0, 1] == self.matrix[0, 2] == self.current_player:
            self.done = True
            return True
        if self.matrix[1, 0] == self.matrix[1, 1] == self.matrix[1, 2] == self.current_player:
            self.done = True
            return True
        if self.matrix[2, 0] == self.matrix[2, 1] == self.matrix[2, 2] == self.current_player:
            self.done = True
            return True
        if self.matrix[0, 0] == self.matrix[1, 0] == self.matrix[2, 0] == self.current_player:
            self.done = True
            return True
        if self.matrix[0, 1] == self.matrix[1, 1] == self.matrix[2, 1] == self.current_player:
            self.done = True
            return True
        if self.matrix[0, 2] == self.matrix[1, 2] == self.matrix[2, 2] == self.current_player:
            self.done = True
            return True
        if self.matrix[0, 0] == self.matrix[1, 1] == self.matrix[2, 2] == self.current_player:
            self.done = True
            return True
        if self.matrix[0, 2] == self.matrix[1, 1] == self.matrix[2, 0] == self.current_player:
            self.done = True
            return True

        return False

    def observation(self):
        return np.array([item for sublist in self.matrix.tolist() for item in sublist])

    def step(self, action):
        if self.done:
            raise Exception('Terminal state, no actions possible')

        if self.matrix[action[0], action[1]] != 0:
            reward = -5

        else:
            self.matrix[action[0], action[1]] = self.current_player
            self.check_done()
            reward = 1 if self.done and 0 in self.matrix else 0
            self.current_player = 2 if self.current_player == 1 else 1

        observation = self.observation()
        done = self.done
        info = 0

        return observation, reward, done, info


def make():
    return Environment()
