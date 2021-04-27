#!/usr/bin/env python
# coding: utf-8

import argparse
import pickle
import numpy as np
from sklearn.decomposition import NMF

NUM_FEATURES = 50


def apply_nmf_on_test(nmf, test_ats, tag):
    base = nmf.components_
    base_weight = nmf.transform(test_ats)
    print("calculating complexities...")
    complexities = (
        np.sqrt(
            np.mean(
                np.square(np.dot(base_weight, base) - test_ats),
                axis=1,
            )
        )
        / np.max(test_ats, axis=1)
    )
    print("done.")

    np.savez(
        "../intermediate_results/nmf_information/{}_base_weight_test".format(tag),
        base_weight,
    )
    np.savez("../results/{}_test".format(tag), complexities)

    return complexities


def train_nmf(target_ats, tag):
    print("fitting NMF...")
    nmf = NMF(
        n_components=NUM_FEATURES,
        init="random",
        beta_loss="kullback-leibler",
        solver="mu",
        max_iter=1000,
    )
    base_weight = nmf.fit_transform(target_ats)
    print("done.")
    base = nmf.components_
    complexities = np.sqrt(
        np.mean(np.square(np.dot(base_weight, base) - target_ats), axis=1)
    ) / np.max(target_ats, axis=1)
    with open(
        "../intermediate_results/nmf_information/{}.pkl".format(tag), mode="wb"
    ) as f:
        pickle.dump(nmf, f)

    np.savez(
        "../intermediate_results/nmf_information/{}_base_weight_train".format(tag),
        base_weight,
    )

    np.savez(
        "../results/{}_train".format(tag),
        complexities,
    )

    return nmf, complexities


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "layer_name",
        help="specify the layer name used to calculate complexities",
        type=str,
    )
    nmf_source = parser.add_mutually_exclusive_group(required=True)
    nmf_source.add_argument(
        "--train",
        help="specify a path of ATs of the training dataset",
        type=str,
        default=None,
    )
    nmf_source.add_argument(
        "--nmf_path",
        help="specify a path of trained NMF information",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--test",
        help="specify a path of ATs of the test dataset",
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

    layer_name = args.layer_name
    train_ats_path = args.train
    test_ats_path = args.test
    nmf_path = args.nmf_path
    tag = args.tag

    if nmf_path is not None:
        with open(nmf_path, mode="rb") as f:
            nmf = pickle.load(f)
            if type(nmf) == dict:
                nmf = nmf[layer_name]
    else:
        print("loading training ATs...")
        train_ats = np.load(train_ats_path)[layer_name]
        nmf, _ = train_nmf(train_ats, tag)

    if test_ats_path is not None:
        print("loading test ATs...")
        test_ats = np.load(test_ats_path)[layer_name]
        apply_nmf_on_test(nmf, test_ats, tag)
