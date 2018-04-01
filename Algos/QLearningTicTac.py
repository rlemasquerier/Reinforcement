import QLearning


class QLearningTicTac(QLearning.QLearning):
    def value(self, state):
        if len(self.q_values(state)) > 0:
            if state.current_player == 1:
                return max(self.q_values(state).values())
            else:
                return min(self.q_values(state).values())
        else:
            return 0.
