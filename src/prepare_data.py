import numpy as np
from lib.datagen import image_brightness, image_saturation, image_contrast
from tqdm import tqdm
import random

from tensorflow.keras.datasets import cifar10


def main():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    np.savez("../datasets/train_dataset_a", x_train, y_train)
    np.savez("../datasets/test_dataset_a", x_test, y_test)
    np.savez(
        "../datasets/train_dataset_b",
        x_train[(y_train != 3) & (y_train != 9)],
        y_train[(y_train != 3) & (y_train != 9)],
    )
    np.savez(
        "../datasets/test_dataset_b",
        x_test[(y_test != 3) & (y_test != 9)],
        y_test[(y_test != 3) & (y_test != 9)],
    )
    print("making dataset_c")
    rand_idx = np.arange(x_train.shape[0])
    random.shuffle(rand_idx)
    bright = np.array([image_brightness(img, 1.4) for img in tqdm(x_train)])
    dark = np.array([image_brightness(img, 0.6) for img in tqdm(x_train)])
    x_target = x_train[rand_idx[16667 * 2 :]]
    x_target = np.append(x_target, bright[rand_idx[:16667]], axis=0)
    x_target = np.append(x_target, dark[rand_idx[16667 : 16667 * 2]], axis=0)
    y_target = y_train[rand_idx[16667 * 2 :]]
    y_target = np.append(y_target, y_train[rand_idx[:16667]], axis=0)
    y_target = np.append(y_target, y_train[rand_idx[16667 : 16667 * 2]], axis=0)
    x_target = x_target[rand_idx]
    y_target = y_target[rand_idx]
    np.savez("../datasets/train_dataset_c", x_target, y_target)

    bright = np.array([image_brightness(img, 1.4) for img in tqdm(x_test)])
    dark = np.array([image_brightness(img, 0.6) for img in tqdm(x_test)])
    np.savez(
        "../datasets/test_dataset_c",
        np.vstack((x_test, bright, dark)),
        np.tile(y_test, (3, 1)),
    )

    print("making dataset_d")
    rand_idx = np.arange(x_train.shape[0])
    random.shuffle(rand_idx)
    high_con = np.array([image_contrast(img, 2.0) for img in tqdm(x_train)])
    low_con = np.array([image_contrast(img, 0.5) for img in tqdm(x_train)])
    x_target = x_train[rand_idx[16667 * 2 :]]
    x_target = np.append(x_target, high_con[rand_idx[:16667]], axis=0)
    x_target = np.append(x_target, low_con[rand_idx[16667 : 16667 * 2]], axis=0)
    y_target = y_train[rand_idx[16667 * 2 :]]
    y_target = np.append(y_target, y_train[rand_idx[:16667]], axis=0)
    y_target = np.append(y_target, y_train[rand_idx[16667 : 16667 * 2]], axis=0)
    x_target = x_target[rand_idx]
    y_target = y_target[rand_idx]
    np.savez("../datasets/train_dataset_d", x_target, y_target)

    high_con = np.array([image_contrast(img, 2.0) for img in tqdm(x_test)])
    low_con = np.array([image_contrast(img, 0.5) for img in tqdm(x_test)])
    np.savez(
        "../datasets/test_dataset_d",
        np.vstack((x_test, high_con, low_con)),
        np.tile(y_test, (3, 1)),
    )

    print("making dataset_e")
    rand_idx = np.arange(x_train.shape[0])
    random.shuffle(rand_idx)
    high_sat = np.array([image_saturation(img, 3.0) for img in tqdm(x_train)])
    low_sat = np.array([image_saturation(img, 0) for img in tqdm(x_train)])
    x_target = x_train[rand_idx[16667 * 2 :]]
    x_target = np.append(x_target, high_sat[rand_idx[:16667]], axis=0)
    x_target = np.append(x_target, low_sat[rand_idx[16667 : 16667 * 2]], axis=0)
    y_target = y_train[rand_idx[16667 * 2 :]]
    y_target = np.append(y_target, y_train[rand_idx[:16667]], axis=0)
    y_target = np.append(y_target, y_train[rand_idx[16667 : 16667 * 2]], axis=0)
    x_target = x_target[rand_idx]
    y_target = y_target[rand_idx]
    np.savez("../datasets/train_dataset_e", x_target, y_target)

    high_sat = np.array([image_saturation(img, 3.0) for img in tqdm(x_test)])
    low_sat = np.array([image_saturation(img, 0) for img in tqdm(x_test)])
    np.savez(
        "../datasets/test_dataset_e",
        np.vstack((x_test, high_sat, low_sat)),
        np.tile(y_test, (3, 1)),
    )


if __name__ == "__main__":
    main()