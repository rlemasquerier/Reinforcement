import numpy as np


class StateTicTacToe:
    def __init__(self, matrix=None, current_player=1):
        if matrix is None:
            self.matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        else:
            self.matrix = matrix
        self.current_player = current_player

    def __hash__(self):
        return hash(self.matrix.tostring()+" "+str(self.current_player))

    def __eq__(self, other):
        return np.array_equal(self.matrix, other.matrix) and self.current_player == other.current_player

    def __copy__(self):
        return StateTicTacToe(np.copy(self.matrix),
                              current_player=self.current_player)

    def __repr__(self):
        return self.matrix.tostring()+" "+str(self.current_player)

    def check_done(self):
        if 0 not in self.matrix:
            return True
        if self.matrix[0, 0] == self.matrix[0, 1] == self.matrix[0, 2] != 0:
            return True
        if self.matrix[1, 0] == self.matrix[1, 1] == self.matrix[1, 2] != 0:
            return True
        if self.matrix[2, 0] == self.matrix[2, 1] == self.matrix[2, 2] != 0:
            return True
        if self.matrix[0, 0] == self.matrix[1, 0] == self.matrix[2, 0] != 0:
            return True
        if self.matrix[0, 1] == self.matrix[1, 1] == self.matrix[2, 1] != 0:
            return True
        if self.matrix[0, 2] == self.matrix[1, 2] == self.matrix[2, 2] != 0:
            return True
        if self.matrix[0, 0] == self.matrix[1, 1] == self.matrix[2, 2] != 0:
            return True
        if self.matrix[0, 2] == self.matrix[1, 1] == self.matrix[2, 0] != 0:
            return True
        return False

    def winner(self):
        if self.matrix[0, 0] == self.matrix[0, 1] == self.matrix[0, 2] != 0:
            return self.matrix[0, 0]
        if self.matrix[1, 0] == self.matrix[1, 1] == self.matrix[1, 2] != 0:
            return self.matrix[1, 0]
        if self.matrix[2, 0] == self.matrix[2, 1] == self.matrix[2, 2] != 0:
            return self.matrix[2, 0]
        if self.matrix[0, 0] == self.matrix[1, 0] == self.matrix[2, 0] != 0:
            return self.matrix[0, 0]
        if self.matrix[0, 1] == self.matrix[1, 1] == self.matrix[2, 1] != 0:
            return self.matrix[0, 1]
        if self.matrix[0, 2] == self.matrix[1, 2] == self.matrix[2, 2] != 0:
            return self.matrix[0, 2]
        if self.matrix[0, 0] == self.matrix[1, 1] == self.matrix[2, 2] != 0:
            return self.matrix[0, 0]
        if self.matrix[0, 2] == self.matrix[1, 1] == self.matrix[2, 0] != 0:
            return self.matrix[0, 2]
        if 0 not in self.matrix:
            return "No Winner"
        return "Not Finished"