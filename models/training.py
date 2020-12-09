import glob

from tensorflow import keras

from data_prep.input_pipeline import get_dataset
from models.model_definition import summed_connection

if __name__ == "__main__":
    stations_path = "../data/stations_data.csv"
    training_paths = glob.glob("../data/data/*/2015-*-*.csv")
    validation_paths = glob.glob("../data/data/*/2016-01-*.csv")

    data = get_dataset(training_paths, stations_path).repeat().shuffle(50).batch(5).prefetch(50)
    validation_data = get_dataset(validation_paths, stations_path).cache().batch(5)

    model = summed_connection((208, 208))
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.RMSprop(),
        metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_categorical_accuracy")],
    )
    model.fit(data, validation_data=validation_data, steps_per_epoch=20, epochs=10)
