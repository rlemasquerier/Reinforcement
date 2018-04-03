# Reinforcement
Some tests for reinforcement learning


## Policy Gradient

### Visualize tensorflow graph in tensorflow

First we need to run a policy gradient training.

Command to visualize graph in tensorboard: 'tensorboard --logdir tf_logs/'

### Save a trained model for production

'python froze_model.py --model_dir C:/YourPath/model --output_node_names multinomial/Multinomial,whateverOtherTFNodeName