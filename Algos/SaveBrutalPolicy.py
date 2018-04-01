from collections import defaultdict
import QLearning
import json


def save_policy(policy, set_states, output_file):
    """

    :param policy:
    :type policy: ```QLearning.PolicyQValue```
    :param setStates:
    :param outputFile:
    :return:
    """
    policy_dict = defaultdict(dict)
    for s in set_states:
        policy_dict[s.__hash__()]['action'] = policy.action(s)
        mat = []
        for l in s.matrix:
            mat += [list(l)]
        policy_dict[s.__hash__()]['state'] = {'matrix': mat, 'player': s.current_player}
    json.dump(policy_dict, open(output_file, 'w'), indent=2)
    return policy_dict