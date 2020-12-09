import glob

from tensorflow import keras

from data_prep.input_pipeline import get_dataset
from models.model_definition import summed_connection

if __name__ == "__main__":
    stations_path = "../data/stations_data.csv"
    training_paths = glob.glob("../data/data/*/2016-06-*.csv")
    validation_paths = glob.glob("../data/data/*/2016-07-*.csv")

    data = get_dataset(training_paths, stations_path).cache().repeat().shuffle(50).batch(5)
    validation_data = get_dataset(validation_paths, stations_path).cache().batch(5)

    model = summed_connection((208, 208))
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(),
        optimizer=keras.optimizers.RMSprop(),
        metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_categorical_accuracy")],
    )
    model.fit(data, validation_data=validation_data, steps_per_epoch=150, epochs=10)
