import QLearning
from collections import defaultdict

class Sarsa(QLearning.PolicyGreedyEpsilon):
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
            a_next = self.action(eo[0])
            q_next = 0. if a_next is None else self.q_values(eo[0])[a_next]
            old_q = self.stored_Q(cur_state, a)
            old_q += self.learningRate * (eo[1] + self.gamma * q_next - old_q)
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