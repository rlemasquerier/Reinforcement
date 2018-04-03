"""

Use the trained policy to play tic tac toe

"""

import tensorflow as tf


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


def play(observation, frozen_model_filename='model/frozen_model.pb'):
    # We use our "load_graph" function
    player_graph = load_graph(frozen_model_filename)

    # We access the input and output nodes
    player_x = player_graph.get_tensor_by_name('prefix/X:0')
    player_y = player_graph.get_tensor_by_name('prefix/multinomial/Multinomial:0')

    # We launch a Session
    with tf.Session(graph=player_graph) as player_sess:
        # Note: we don't nee to initialize/restore anything
        # There is no Variables in this graph, only hardcoded constants
        y_out = player_sess.run(player_y, feed_dict={
            player_x: observation.reshape(1, player_x.shape[1])
        })

    return int(y_out[0][0])
