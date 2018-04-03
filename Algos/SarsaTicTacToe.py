import SarsaLearning


class SarsaTicTac(SarsaLearning.Sarsa):
    def value(self, state):
        print "Hey "
        if len(self.q_values(state)) > 0:
            if state.current_player == 1:
                return max(self.q_values(state).values())
            else:
                return min(self.q_values(state).values())
        else:
            return None