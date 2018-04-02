import DomainTicTacToe
import StateTicTacToe
import EnvironmentTicTacToe
import tic_tac_toe
import QLearning
import json
import random

initialState = StateTicTacToe.StateTicTacToe()
domainTicTac = DomainTicTacToe.DomainTicTacToe()
environementTicTacToe = EnvironmentTicTacToe.EnvironmentTicTacToe(domainTicTac, initial_state=initialState)
policySarsa = json.load(open('policy_tic_tac_sarsa_2.json', 'r'))
policyQLearning = json.load(open('policy_tic_tac_2.json', 'r'))


class PolicyTicTacFromDict(QLearning.PolicyQValue):
    def __init__(self, policyDict):
        self.policyDict = policyDict

    def action(self, state):
        if str(state.__hash__()) in self.policyDict:
            return tuple(self.policyDict[str(state.__hash__())]['action'])
        else:
            b = tic_tac_toe.ActionSpace(state.matrix)
            ac = b.space
            return random.choice(tuple(ac))

sarsa = PolicyTicTacFromDict(policySarsa)
qlearning = PolicyTicTacFromDict(policyQLearning)
nb_games = 10000
winner = {"sarsa":0, "qlearning":0}
for j in range(nb_games):
    player_sarsa = 2
    player_qlearning = 1
    if j > nb_games/2:
        player_qlearning=2
        player_sarsa=1
    environementTicTacToe.initial_state = initialState
    cur_state = environementTicTacToe.current_observation()
    first_move=True
    print cur_state
    while not environementTicTacToe.is_terminal():
        if first_move:
            ac = domainTicTac.available_actions(cur_state)
            action = random.choice(tuple(ac))
            first_move = False
        elif player_sarsa == cur_state.current_player:
            action = sarsa.action(cur_state)
        else:
            action = qlearning.action(cur_state)
        cur_state, reward, done, info = environementTicTacToe.execute_action(action)
        if done:
            print "Game finished ", info
            print cur_state.matrix
            if info != "":
                print "Sarsa won" if player_qlearning == cur_state.current_player else "QLearning Won"
                if player_qlearning == cur_state.current_player:
                    winner['sarsa'] += 1
                else:
                    winner['qlearning'] += 1
            else:
                print "Draw "
    print winner




