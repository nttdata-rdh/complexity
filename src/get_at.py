import argparse
import numpy as np
from tqdm import tqdm

from tensorflow.keras import Model
from tensorflow.keras.models import load_model

BATCH_SIZE = 5000


def get_at(model, data, tag, layer_name=None):
    current_at = {}
    for layer in tqdm(model.layers):
        if layer_name is not None and layer_name != layer.name:
            continue
        if "activation" not in layer.name:
            continue
        intermediate_model = Model(model.input, layer.output)
        if data.shape[0] > 5000:
            current_at[layer.name] = None
            for i in tqdm(range(data.shape[0] // BATCH_SIZE)):
                tmp_data = data[i * BATCH_SIZE : (i + 1) * BATCH_SIZE, ...]
                current_at[layer.name] = (
                    intermediate_model.predict(tmp_data).reshape(tmp_data.shape[0], -1)
                    if current_at[layer.name] is None
                    else np.append(
                        current_at[layer.name],
                        intermediate_model.predict(tmp_data).reshape(
                            tmp_data.shape[0], -1
                        ),
                        axis=0,
                    )
                )
        else:
            current_at[layer.name] = intermediate_model.predict(data).reshape(
                data.shape[0], -1
            )
    np.savez_compressed(
        "../intermediate_results/activation_traces/{}".format(tag), **current_at
    )
    return current_at


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="specify a path of the trained model", type=str)
    parser.add_argument(
        "dataset",
        help="specify a path of the dataset that you want to encode",
        type=str,
    )
    parser.add_argument(
        "--tag",
        help="specify a tag name to identify the output",
        type=str,
        default="activation_trace",
    )
    parser.add_argument(
        "--layer",
        help="specify a layer name used to obtain ats",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    model = load_model(args.model)
    dataset = np.load(args.dataset)["arr_0"]
    tag = args.tag
    layer = args.layer
    get_at(model, dataset, tag, layer)
