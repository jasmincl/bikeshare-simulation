from typing import List

import graph_nets
import tensorflow as tf
from graph_nets.graphs import GraphsTuple
from tensorflow import keras

keras.backend.set_floatx("float64")


def get_graph_net_keras_inputs(
    node_feature_dim: int, edge_feature_dim: int, global_feature_dim: int
) -> List[keras.Input]:
    return [
        keras.Input([None, node_feature_dim], dtype=tf.float64),
        keras.Input([None, edge_feature_dim], dtype=tf.float64),
        keras.Input([None], dtype=tf.int64),
        keras.Input([None], dtype=tf.int64),
        keras.Input([global_feature_dim], dtype=tf.float64),
    ]


class GraphNetWrapper(keras.layers.Layer):
    def __init__(self, graph_net: graph_nets.modules.GraphNetwork):
        super(GraphNetWrapper, self).__init__(dynamic=True, autocast=False)
        self.graph_net = graph_net

    @property
    def graph_order_names(self) -> List[str]:
        return ["nodes", "edges", "senders", "receivers", "globals"]

    def _get_tensor_array(self, graph_tuple: GraphsTuple) -> List[tf.Tensor]:
        result = [getattr(graph_tuple, name) for name in self.graph_order_names]
        result[-1] = result[-1][0]
        return result

    @staticmethod
    def _get_graph_tuple(input_tensors: List[tf.Tensor]) -> GraphsTuple:
        return GraphsTuple(
            nodes=input_tensors[0],
            edges=input_tensors[1],
            senders=input_tensors[2],
            receivers=input_tensors[3],
            globals=tf.expand_dims(input_tensors[4], axis=0),
            n_node=tf.convert_to_tensor([input_tensors[0].shape[0]]),
            n_edge=tf.convert_to_tensor([input_tensors[1].shape[0]]),
        )

    def _call_graph(self, input_tensors: List[tf.Tensor]) -> List[tf.Tensor]:
        graph = self._get_graph_tuple(input_tensors)
        output_graph = self.graph_net(graph)
        return self._get_tensor_array(output_graph)

    @staticmethod
    def _get_example_tensor_list(
        number_features_nodes: int,
        number_features_edges: int,
        number_features_global: int,
    ) -> List[tf.Tensor]:
        return [
            tf.zeros([1, number_features_nodes], dtype=tf.double),
            tf.zeros([1, number_features_edges], dtype=tf.double),
            tf.convert_to_tensor([0]),
            tf.convert_to_tensor([0]),
            tf.zeros([number_features_global], dtype=tf.double),
        ]

    def build(self, input_shape):
        example_inputs = self._get_example_tensor_list(
            number_features_nodes=input_shape[0][-1],
            number_features_edges=input_shape[1][-1],
            number_features_global=input_shape[4][-1],
        )
        self._call_graph(example_inputs)

    def call(self, inputs, **kwargs):
        return tf.map_fn(self._call_graph, inputs)

    def compute_output_shape(self, input_shape):
        example_inputs = self._get_example_tensor_list(
            number_features_nodes=input_shape[0][-1],
            number_features_edges=input_shape[1][-1],
            number_features_global=input_shape[4][-1],
        )
        example_output = self._call_graph(example_inputs)
        return [
            (None, None, example_output[0].shape[-1]),
            (None, None, example_output[1].shape[-1]),
            (None, None),
            (None, None),
            (None, example_output[4].shape[-1]),
        ]
