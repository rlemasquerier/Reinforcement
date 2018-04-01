import tic_tac_toe
import StateTicTacToe


class DomainTicTacToe(tic_tac_toe.Environment):
    def reset(self):
        tic_tac_toe.Environment.reset(self)

    def __init__(self):
        tic_tac_toe.Environment.__init__(self)

    def available_actions(self):
        return tic_tac_toe.Environment.available_actions(self)

    def step(self, state, action):
        """

        :param state:
        :type state: ```StateTicTacToe.StateTicTacToe```
        :param action:
        :return:
        """
        new_state = state.__copy__()
        if self.matrix[action[0], action[1]] != 0:
            reward = -5
        else:
            new_state.matrix[action[0], action[1]] = new_state.current_player
            finished_or_winner = new_state.winner()
            reward = 0.
            done = False
            if finished_or_winner == "Not Finished":
                done = False
            else:
                done = True
                if finished_or_winner == 1:
                    reward = 1.
                elif finished_or_winner == 2:
                    reward = -1.
                else:
                    reward = 0.
            new_state.current_player = 2 if new_state.current_player == 1 else 1
        info = ""
        if reward == 1.:
            info = "Winner Player One "
        if reward == -1.:
            info = "Winner Player Two"
        return new_state, reward, done, info

    def available_actions(self, state):
        """
        :param state:
        :type state: ```StateTicTacToe.StateTicTacToe```
        :param action:
        :return:
        """
        return tic_tac_toe.ActionSpace(state.matrix).space

