# Reinforcement

[![Join the chat at https://gitter.im/reinforcement_community/Lobby](https://badges.gitter.im/reinforcement_community/Lobby.svg)](https://gitter.im/reinforcement_community/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

The goal of this repo is to experiment some Reinforcement Leanring algorithms on customs and simple games.

## Policy Gradient

### Algorithm overview

A policy gradient algorithm is used in the context of neural network policy. The neural network take an observation as an input and the output is the action to be executed. To be more specific, the output is a probability to take one action of the action space. This is to let the agent have a right balance between exploring new states and also using actions that are known to work well.

Now, how can we train the neural network so that the policy takes the best decision at each step ?

### Evaluate an action 

### Visualize tensorflow graph in tensorflow

First we need to run a policy gradient training.

Command to visualize graph in tensorboard: `tensorboard --logdir tf_logs/`

### Save a trained model for production

`python froze_model.py --model_dir C:/YourPath/model --output_node_names multinomial/Multinomial,whateverOtherTFNodeName`
