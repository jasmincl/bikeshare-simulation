import graph_nets
import sonnet
from tensorflow import keras
import tensorflow as tf

from models.graph_net_wrapper import get_graph_net_keras_inputs, GraphNetWrapper


def summed_connection(inp_shape) -> keras.Model:
    """
    Model to sum every connection per start point and using a ff network
    Predict which hour of the day features belong to
    """
    input_layer = keras.Input(inp_shape)
    dense = keras.layers.Dense(100, activation="relu")(input_layer)
    output = keras.layers.Dense(24, activation="softmax")(dense)

    return keras.Model(inputs=input_layer, outputs=output)


def graph_model(
    node_feature_dim: int, edge_feature_dim: int, global_feature_dim: int
) -> keras.Model:
    inputs = get_graph_net_keras_inputs(
        node_feature_dim, edge_feature_dim, global_feature_dim
    )
    global_block_opt = {"use_nodes": True, "use_globals": True, "use_edges": True}
    node_block_opt = {
        "use_nodes": True,
        "use_globals": True,
        "use_received_edges": True,
    }
    edge_block_opt = {
        "use_receiver_nodes": True,
        "use_sender_nodes": True,
        "use_globals": True,
    }
    graph_net = graph_nets.modules.GraphNetwork(
        edge_model_fn=lambda: sonnet.nets.MLP([50]),
        node_model_fn=lambda: sonnet.nets.MLP([50]),
        global_model_fn=lambda: sonnet.nets.MLP([24]),
        edge_block_opt=edge_block_opt,
        node_block_opt=node_block_opt,
        global_block_opt=global_block_opt,
        reducer=tf.math.unsorted_segment_mean,
    )
    graph_net_layer = GraphNetWrapper(graph_net)
    graph_output = graph_net_layer(inputs)

    global_features = graph_output[-1]
    output_layer = keras.layers.Activation("softmax")(global_features)

    return keras.Model(inputs=inputs, outputs=output_layer)
