import numpy as np
from tensorflow import keras

from models.model_definition import graph_model

if __name__ == "__main__":
    # ToDo: Write input data pipeline and train Model with real data
    model = graph_model(node_feature_dim=2, edge_feature_dim=3, global_feature_dim=5)

    model.compile(
        loss=keras.losses.CategoricalCrossentropy(),
        optimizer=keras.optimizers.RMSprop(),
        metrics=["accuracy"],
    )
    test_data = [
        np.random.sample([100, 50, 2]),
        np.random.sample([100, 100, 3]),
        np.random.randint(low=0, high=49, size=[100, 100]),
        np.random.randint(low=0, high=49, size=[100, 100]),
        np.random.sample([100, 5]),
    ]
    test_labels = np.zeros((100, 24))
    for i in range(100):
        test_labels[i][np.random.randint(low=0, high=23, size=1)[0]] = 1

    model.fit(test_data, test_labels, epochs=10, batch_size=5)
