import DomainTicTacToe
import StateTicTacToe
import EnvironmentTicTacToe
import tic_tac_toe
import QLearning
import json


initialState = StateTicTacToe.StateTicTacToe()
domainTicTac = DomainTicTacToe.DomainTicTacToe()
environementTicTacToe = EnvironmentTicTacToe.EnvironmentTicTacToe(domainTicTac, initial_state=initialState)
policyDict = json.load(open('policy_tic_tac.json', 'r'))

class PolicyTicTacFromDict(QLearning.PolicyQValue):
    def __init__(self, policyDict):
        self.policyDict = policyDict

    def action(self, state):
        return tuple(self.policyDict[str(state.__hash__())]['action'])

finalPolicy = PolicyTicTacFromDict(policyDict)
environementTicTacToe.initial_state = initialState
player_human = 1 # You play first, or 2 : you play second
cur_state = environementTicTacToe.current_observation()
while not environementTicTacToe.is_terminal():
    if player_human == cur_state.current_player:
        print "It's your turn... Current table :"
        print cur_state.matrix
        print "Choose : "
        ac = list(environementTicTacToe.domain.available_actions(cur_state))
        for i in range(len(ac)):
            print "i : ", i, " : ", ac[i]
        choice = raw_input()
        try:
            choice = int(choice)
            if choice >= len(ac):
                print "Noob"
        except ValueError:
            print "Noobie !!"
        action = ac[choice]
    else:
        action = finalPolicy.action(cur_state)
    cur_state, reward, done, info = environementTicTacToe.execute_action(action)
    if done:
        print "Game finished ", info
        print cur_state.matrix




