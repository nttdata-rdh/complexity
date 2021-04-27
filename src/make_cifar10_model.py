import argparse
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Activation,
    Dropout,
    Flatten,
)
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras import utils

INPUT_SHAPE = (32, 32, 3)  # cifar-10
NUM_CLASSES = 10


def build_model():
    model = Sequential()

    model.add(
        Conv2D(
            32,
            (3, 3),
            padding="same",
            input_shape=INPUT_SHAPE,
        )
    )
    model.add(
        Activation(
            "relu",
        )
    )
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(
        Activation(
            "relu",
        )
    )
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1024, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(
        Activation(
            "relu",
        )
    )
    model.add(Dropout(0.5))
    model.add(Dense(512, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(
        Activation(
            "relu",
        )
    )
    model.add(Dropout(0.5))
    model.add(
        Dense(
            10,
        )
    )

    model.add(
        Activation(
            "softmax",
        )
    )
    model.compile(
        loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"]
    )

    return model


def load_data(dataset_path):
    x, y = np.load(dataset_path).values()
    return ((x, utils.to_categorical(y, num_classes=NUM_CLASSES)),)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_dataset", help="specify a path of a training dataset", type=str
    )
    parser.add_argument(
        "valid_dataset", help="specify a path of a validation dataset", type=str
    )
    parser.add_argument(
        "--tag",
        help="specify a tag name to identify the output",
        type=str,
        default="new_model",
    )
    args = parser.parse_known_args()

    model = build_model()
    (X_train, y_train) = load_data(args.train_dataset)
    (X_valid, y_valid) = load_data(args.valid_dataset)
    print(X_train.shape, y_train.shape)
    print(X_valid.shape, y_valid.shape)
    model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=128,
        verbose=2,
        validation_data=(X_valid, y_valid),
    )
    model.save("../trained_model/{}.h5".format(args.tag))
