import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import os
import math
import platform
import pickle

def load_CIFAR_batch(file_path):
    """
    Load single batch of CIFAR-10 images from
    the binary file and return as a NumPy array.
    """
    with open(file_path, "rb") as f:
        data_dict = pickle.load(f, encoding="latin1")

        # Extract NumPy from dictionary.
        X = data_dict["data"]
        y = data_dict["labels"]

        # Reshape and transpose flat array as 32 X 32 RGB image.
        X = X.reshape(CIFAR_BATCH_SIZE, *input_shape, order="F")
        X = X.transpose((0, 2, 1, 3))

        # Convert `labels` to vector.
        y = np.expand_dims(y, axis=1)

        return X, y


def load_CIFAR10(cv_size=0.25):
    """
    Load all batches of CIFAR-10 images from the
    binary file and return as TensorFlow DataSet.
    """
    X_btchs = []
    y_btchs = []
    for batch in range(1, 6):
        file_path = os.path.join(ROOT, "data_batch_%d" % (batch,))
        X, y = load_CIFAR_batch(file_path)
        X_btchs.append(X)
        y_btchs.append(y)

    # Combine all batches.
    all_Xbs = np.concatenate(X_btchs)
    all_ybs = np.concatenate(y_btchs)

    # Convert Train dataset from NumPy array to TensorFlow Dataset.
    ds_all = tf.data.Dataset.from_tensor_slices((all_Xbs, all_ybs))

    al_size = len(ds_all)
    tr_size = math.ceil((1 - cv_size) * al_size)
    # Split dataset into Train and Cross-validation sets.
    ds_tr = ds_all.take(tr_size)
    ds_cv = ds_all.skip(tr_size)
    print(f"Train dataset size: {tr_size}.")
    print(f"Cross-validation dataset size: {al_size - tr_size}.")

    # Convert Test dataset from NumPy array to TensorFlow Dataset.
    X_ts, y_ts = load_CIFAR_batch(os.path.join(ROOT, "test_batch"))
    ds_ts = tf.data.Dataset.from_tensor_slices((X_ts, y_ts))
    print(f"Test dataset size {len(ds_ts)}.")

    return ds_tr, ds_cv, ds_ts

ROOT = "../cifar-10-batches-py/"

CIFAR_BATCH_SIZE = 10000
img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols, 3)

#ds_tr, ds_cv, ds_ts = load_CIFAR10()
