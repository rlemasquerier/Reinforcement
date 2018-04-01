import DomainTicTacToe
import StateTicTacToe
import EnvironmentTicTacToe
import QLearning


def rollout(policy, environement):
    """
    :type policy: ```QLearning.PolicyQValue```
    :param environement:
    :type environement: ```EnvironmentTicTacToe.EnvironmentTicTacToe```
    :return:
    """
    episode = []
    while not environement.is_terminal():
        curstate = environement.current_observation()
        a = policy.action(curstate)
        eo = environement.execute_action(a)  # state, reward, done, info
        episode += [(a, eo)]
    return episode


def render_episode(episode):
    for i in range(len(episode)):
        print 'i : ', i
        print 'Player : ', 1 if episode[i][1][0].current_player == 2 else 2
        print 'Action : ', episode[i][0]
        print 'State : '
        print episode[i][1][0].matrix

class PolicyGreedyTwoPlayer(QLearning.PolicyQValue):
    def __init__(self, algo):
        self.algo = algo

    def action(self, state):
        qvalues = self.algo.q_values(state)
        if state.current_player == 1:
            print "player 1"
            print qvalues
            return max(qvalues, key=lambda x: qvalues[x])
        else:
            print "player 2"
            print qvalues
            return min(qvalues, key=lambda x: qvalues[x])


initialState = StateTicTacToe.StateTicTacToe()
domainTicTac = DomainTicTacToe.DomainTicTacToe()
environementTicTacToe = EnvironmentTicTacToe.EnvironmentTicTacToe(domainTicTac, initial_state=initialState)

def v(s, a):
    return -2

algo = QLearning.QLearning(domain=domainTicTac,
                           gamma=1.,
                           qinit=v,
                           learningRate=0.1,
                           epsilon=0.1)

from tqdm import tqdm
for i in tqdm(range(1000000)):
    algo.run_learning_episode(environment=environementTicTacToe, maxSteps=10)
    environementTicTacToe = EnvironmentTicTacToe.EnvironmentTicTacToe(domainTicTac, initial_state=initialState)

finalPolicy = PolicyGreedyTwoPlayer(algo)
environementTicTacToe = EnvironmentTicTacToe.EnvironmentTicTacToe(domainTicTac, initial_state=initialState)
episode = rollout(finalPolicy, environementTicTacToe)
render_episode(episode)

# Play against the IA...
environementTicTacToe = EnvironmentTicTacToe.EnvironmentTicTacToe(domainTicTac,
                                                                  initial_state=initialState)
player_human = 2
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











