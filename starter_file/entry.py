import argparse
from azureml.core.run import Run
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

run = Run.get_context()

# Read data from csv
X = pd.read_csv(r"./../../data/heart_failure_clinical_records_dataset.csv")
Y = X.pop("DEATH_EVENT")

# Split data into train and test sets.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help="The rate at which the optimizer updates weights")
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of times of training the model")
    parser.add_argument('--neurons', type=int, default=12,
                        help="Number of units (neurons) inside the hidden layer")

    args = parser.parse_args()

    run.log("learning_rate:", np.float(args.learning_rate))
    run.log("epochs:", np.float(args.epochs))
    run.log("neurons:", np.float(args.neurons))

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(12,)),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(units=args.neurons, activation=tf.nn.relu, kernel_regularizer='l2'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(units=1, activation=tf.nn.sigmoid)
    ])

    # Create the model architecture
    opt = keras.optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(optimizer=opt,  # Generate new guess about the input/output relation
                  loss='binary_crossentropy',  # Calculate how good the guess was
                  metrics=['accuracy'])

    model.fit(X_train, Y_train, epochs=args.epochs)

    # Then
    test_loss, test_acc = model.evaluate(X_test, Y_test)
    print('Test accuracy:', test_acc)

    model.save(
        'my_model' + '_LR_' + str(args.learning_rate) + '_epochs_' + str(args.epochs) + '_units_' + str(args.neurons))

    run.log("accuracy", np.float(test_acc))


main()
