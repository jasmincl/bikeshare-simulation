import glob

from tensorflow import keras

from models.model_definition import graph_model
from pipeline.graph_net import (
    get_dataset,
    NODE_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
)

if __name__ == "__main__":
    stations_path = "../data/stations_data.csv"
    training_paths = [f"../data/pickle_data/2016-{i:02}.pickle" for i in range(1, 7)]
    validation_paths = glob.glob("../data/pickle_data/2016-07.pickle")

    data = get_dataset(training_paths, stations_path).shuffle(50).batch(1)
    validation_data = get_dataset(validation_paths, stations_path).cache().batch(1)

    model = graph_model(
        node_feature_dim=NODE_FEATURE_DIM,
        edge_feature_dim=EDGE_FEATURE_DIM,
        global_feature_dim=GLOBAL_FEATURE_DIM,
    )

    model.compile(
        loss=keras.losses.CategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy"],
    )

    model.fit(data, validation_data=validation_data, epochs=10)
    while True:
        model.fit(data, epochs=1, steps_per_epoch=100)
    model.fit(data, epochs=1, steps_per_epoch=100)
