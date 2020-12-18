import graph_nets
import sonnet
from tensorflow import keras

from models.graph_net_wrapper import get_graph_net_keras_inputs, GraphNetWrapper


def sum_layer(tensor):
    return keras.backend.sum(tensor, axis=-1)


def summed_connection(inp_shape) -> keras.Model:
    """
    Model to sum every connection per start point and using a ff network
    Predict which hour of the day features belong to
    """
    input_layer = keras.Input(inp_shape)
    dense = keras.layers.Lambda(sum_layer)(input_layer)
    dense = keras.layers.Dense(100, activation="relu")(dense)
    output = keras.layers.Dense(24, activation="softmax")(dense)

    return keras.Model(inputs=input_layer, outputs=output)


def graph_model(
    node_feature_dim: int, edge_feature_dim: int, global_feature_dim: int
) -> keras.Model:
    inputs = get_graph_net_keras_inputs(
        node_feature_dim, edge_feature_dim, global_feature_dim
    )
    global_block_opt = {"use_nodes": True, "use_globals": False}
    node_block_opt = {"use_nodes": True, "use_globals": False}
    edge_block_opt = {
        "use_receiver_nodes": True,
        "use_sender_nodes": True,
        "use_globals": False,
    }
    graph_net = graph_nets.modules.GraphNetwork(
        edge_model_fn=lambda: sonnet.nets.MLP([10, 5]),
        node_model_fn=lambda: sonnet.nets.MLP([10, 5]),
        global_model_fn=lambda: sonnet.nets.MLP([10, 50]),
        edge_block_opt=edge_block_opt,
        node_block_opt=node_block_opt,
        global_block_opt=global_block_opt,
    )
    graph_net_layer = GraphNetWrapper(graph_net)
    graph_output = graph_net_layer(inputs)
    global_features = graph_output[-1]
    global_features = keras.layers.Dropout(0.1)(global_features)
    output = keras.layers.Dense(24, activation="softmax")(global_features)
    return keras.Model(inputs=inputs, outputs=output)
