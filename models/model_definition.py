from tensorflow import keras


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
