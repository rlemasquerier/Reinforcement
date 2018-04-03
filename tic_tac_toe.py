"""

"""

import numpy as np
import random
import player


POLICY_GRADIENT_MODEL_PATH = 'saved_models/frozen_model_1.pb'


def _to_action_space(integer_action):
    return int(integer_action / 3), integer_action % 3


class ActionSpace:
    def __init__(self, matrix):
        self.space = set()
        for i in range(3):
            for j in range(3):
                if matrix[i, j] == 0:
                    self.space.add((i, j))

    def __contains__(self, item):
        return item in self.space

    def __len__(self):
        return len(self.space)

    def remove(self, action):
        if action in self.space:
            self.space.remove(action)

    def sample(self):
        return random.sample(self.space, 1)[0]


class Environment:
    def __init__(self):
        self.matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.action_space = ActionSpace(self.matrix)
        self.current_player = 1
        self.done = False
        self.victory = False

    def reset(self):
        self.matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.action_space = ActionSpace(self.matrix)
        self.current_player = 1
        self.done = False
        self.victory = False
        return self.observation()

    def render(self):
        print(self.matrix)

    def check_victory(self):
        if self.matrix[0, 0] == self.matrix[0, 1] == self.matrix[0, 2] == self.current_player:
            return True
        if self.matrix[1, 0] == self.matrix[1, 1] == self.matrix[1, 2] == self.current_player:
            return True
        if self.matrix[2, 0] == self.matrix[2, 1] == self.matrix[2, 2] == self.current_player:
            return True
        if self.matrix[0, 0] == self.matrix[1, 0] == self.matrix[2, 0] == self.current_player:
            return True
        if self.matrix[0, 1] == self.matrix[1, 1] == self.matrix[2, 1] == self.current_player:
            return True
        if self.matrix[0, 2] == self.matrix[1, 2] == self.matrix[2, 2] == self.current_player:
            return True
        if self.matrix[0, 0] == self.matrix[1, 1] == self.matrix[2, 2] == self.current_player:
            return True
        if self.matrix[0, 2] == self.matrix[1, 1] == self.matrix[2, 0] == self.current_player:
            return True
        return False

    def check_done(self):
        if len(self.action_space) == 0:
            return True
        return False

    def observation(self):
        return np.array([item for sublist in self.matrix.tolist() for item in sublist])

    def step(self, action):
        if type(action) is int:
            action = _to_action_space(action)
        if action not in self.action_space:
            print(self.matrix)
            print(action)
            raise Exception('Action is not possible')
        if self.done:
            print('last action', action)
            print(self.matrix)
            raise Exception('Terminal state, no actions possible')

        else:
            self.matrix[action[0], action[1]] = self.current_player
            self.action_space.remove(action)
            self.victory = self.check_victory()
            self.done = self.check_done() or self.victory
            self.current_player = -1 if self.current_player == 1 else 1
            reward = 0

        observation = self.observation()
        done = self.done
        info = 0
        return observation, reward, done, info

    def available_actions(self):
        return ActionSpace(self.matrix).space


class EnvironmentSimulation(Environment):
    def __init__(self, opponent='random', begin=True):
        Environment.__init__(self)
        self.opponent = opponent
        self.begin = begin
        self.player = player.Player()
        if opponent == 'policy_gradient':
            self.player.initialize_model(POLICY_GRADIENT_MODEL_PATH)

    def reset(self):
        Environment.reset(self)
        if not self.begin:
            initial_play = (random.randint(0, 2), random.randint(0, 2))
            self.matrix[initial_play] = -1
            self.action_space.remove(initial_play)
        return Environment.observation(self)

    def step(self, action):
        observation, reward, done, info = Environment.step(self, action)
        if self.victory:
            reward = 5
            return observation, reward, done, info

        if self.done:
            return observation, reward, done, info

        if self.opponent == 'random':
            action = self.player.play(observation)
        elif self.opponent == 'policy_gradient':
            action = self.player.play(observation)
        else:
            raise Exception('opponent mode not correct')

        observation, reward, done, info = Environment.step(self, action)

        if self.victory:
            reward = -5
            return observation, reward, done, info

        if self.done:
            return observation, reward, done, info

        return observation, reward, done, info


def make():
    return Environment()


def make_simulation(opponent='random', begin=True):
    return EnvironmentSimulation(opponent, begin)
