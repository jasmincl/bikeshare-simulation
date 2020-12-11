import glob

from tensorflow import keras

from data_prep.input_pipeline_pickle import get_dataset
from models.model_definition import summed_connection

number_stations = 208

if __name__ == "__main__":
    stations_path = "../data/stations_data.csv"
    training_paths = [f"../data/pickle_data/2016-{i:02}.pickle" for i in range(1, 7)]
    validation_paths = glob.glob("../data/data/*/2016-07.pickle")

    data = get_dataset(training_paths, number_stations).shuffle(50).batch(5)
    validation_data = get_dataset(validation_paths, number_stations).cache().batch(5)

    model = summed_connection((number_stations, number_stations))
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(),
        optimizer=keras.optimizers.RMSprop(),
        metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_categorical_accuracy")],
    )
    model.fit(data, validation_data=validation_data, epochs=50)
