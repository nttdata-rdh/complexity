import argparse
import os

import numpy as np
from tensorflow.keras.models import load_model

from src.get_at import get_at
from src.train_nmf import train_nmf, apply_nmf_on_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_path",
        help="specify a path of the trained model",
        type=str,
    )
    parser.add_argument(
        "train_dataset_path",
        help="specify a path of a training dataset",
        type=str,
    )
    parser.add_argument(
        "layer_name",
        help="specify the layer name used to calculate complexities",
        type=str,
    )
    parser.add_argument(
        "--test_dataset_path",
        help="specify a path of a test dataset",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--tag",
        help="specify a tag name to identify the output",
        type=str,
        default="results",
    )
    args = parser.parse_args()

    model = load_model(args.model_path)
    layer_name = args.layer_name
    train_dataset = np.load(args.train_dataset_path)["arr_0"]
    test_dataset = (
        None
        if args.test_dataset_path is None
        else np.load(args.test_dataset_path)["arr_0"]
    )
    tag = args.tag

    os.chdir("./src")
    train_ats = get_at(model, train_dataset, tag, layer_name=layer_name)[layer_name]
    nmf, train_complexities = train_nmf(train_ats, tag)
    if test_dataset is not None:
        test_ats = get_at(model, test_dataset, tag, layer_name=layer_name)[layer_name]
        test_complexities = apply_nmf_on_test(nmf, test_ats, tag)
