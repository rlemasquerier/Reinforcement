"""

Helpers functions

"""

import numpy as np


def discount_rewards(rewards, discount_rate):
    """

    :param rewards: The rewards to discount
    :param discount_rate: The discount rate to compute discounted rewards
    :return: discounted rewards

    Exemple : rewards = [10, 2, -8]
              discount_rate = 0.5
              return [10+0.5*2+0.25*-8, 2+0.5*-8, -8] = [9, -2, -8]
    """
    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards


def discount_and_normalize_rewards(all_rewards, discount_rate):
    """

    :param all_rewards: a list with all rewards obtained per game
    :param discount_rate: the discount rate
    :return: discount all rewards lists using previous function, and normalize it
    """
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]

