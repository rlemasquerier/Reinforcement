from collections import defaultdict
import random

class QProvider:
    def q_values(self, state):
        raise NotImplementedError("You should implement !")


class PolicyQValue(QProvider):
    def action(self, state):
        raise NotImplementedError("You should implement !")


class PolicyGreedy(PolicyQValue):
    def __init__(self, algo):
        self.algo = algo

    def action(self, state):
        qvalues = self.algo.q_values(state)
        if len(qvalues) > 0:
            return max(qvalues, key=lambda x: qvalues[x])
        else:
            return None


class PolicyGreedyEpsilon(PolicyQValue):
    def __init__(self, domain, epsilon=0.1):
        self.epsilon = epsilon
        self.domain = domain

    def action(self, state):
        d = random.random()
        if d < self.epsilon:
            acs = self.domain.available_actions(state)
            if len(acs) > 0:
                return random.choice(tuple(acs))
            return None
        qvalues = self.q_values(state)
        if len(qvalues) > 0:
            if state.current_player == 1:
                return max(qvalues, key=lambda x: qvalues[x])
            else:
                return min(qvalues, key=lambda x: qvalues[x])
        else:
            return 0.


class QLearning(PolicyGreedyEpsilon):
    def __init__(self, domain, gamma,
                 qinit, learningRate, epsilon):
        self.domain = domain
        self.gamma = gamma
        self.qinit = qinit
        if self.qinit is None:
            def qinit(*_):
                return 0.
            self.qinit = qinit
        self.qvalues = defaultdict()
        self.learningRate = learningRate
        self.epsilon = epsilon

    def run_learning_episode(self, environment, maxSteps):
        """

        :param environement:
        :type environement: ```EnvironmentTicTacToe.EnvironmentTicTacToe```
        :param maxSteps:
        :return:
        """
        e = environment.current_observation()
        episode = [e]
        cur_state = e
        steps = 0
        while not environment.is_terminal() and (steps < maxSteps or maxSteps == -1):
            a = self.action(cur_state)
            eo = environment.execute_action(a) # state, reward, done, info
            episode += [eo]
            max_q = self.value(eo[0])
            old_q = self.stored_Q(cur_state, a)
            old_q += self.learningRate * (eo[1] + self.gamma * max_q - old_q)
            self.qvalues[cur_state][a] = old_q
            cur_state = eo[0]
            steps += 1

    def q_values(self, state):
        if state in self.qvalues:
            qs = self.qvalues[state]
        else:
            actions = self.domain.available_actions(state)
            qs = {ac: self.qinit(state, ac) for ac in actions}
            self.qvalues[state] = qs
        return qs

    def value(self, state):
        if len(self.q_values(state)) > 0:
            return max(self.q_values(state).values())
        else:
            return 0.

    def stored_Q(self, state, action):
        return self.q_values(state)[action]

