import StateTicTacToe
import DomainTicTacToe

class EnvironmentTicTacToe:
    def __init__(self, domain, initial_state):
        """

        :param domain:
        :type domain: ```DomainTicTacToe.DomainTicTacToe```
        :type initial_state: ```StateTicTacToe.StateTicTacToe```
        """
        self.domain = domain
        self.initial_state = initial_state
        if self.initial_state is None:
            self.initial_state = StateTicTacToe.StateTicTacToe()

    def current_observation(self):
        return self.initial_state

    def is_terminal(self):
        return self.initial_state.check_done()

    def execute_action(self, action):
        new_state, reward, done, info = self.domain.step(self.initial_state, action)
        self.initial_state = new_state
        return new_state, reward, done, info
