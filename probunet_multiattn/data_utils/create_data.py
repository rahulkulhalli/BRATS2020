import os
import shutil
import pandas as pd
import numpy as np


def reset_df_index(df):
        return df.reset_index(drop=True, inplace=False)


def prepare_data(csv_path, train_split=0.8, seed=None):
    """
    An internal method to partition data into train and test based
    on the ratio provided in __init__.
    """
    
    if seed:
        np.random.seed(seed)
        print("Seed set.")

    # First, read the input CSV.
    if not os.path.exists(csv_path):
        raise FileNotFoundError("Input CSV not found.")
    
    # We really only need these two columns to work with.
    df = pd.read_csv(csv_path, usecols=["Grade", "BraTS_2020_subject_ID"])

    hgg_data = df.loc[df.Grade == "HGG", :]
    hgg_data = reset_df_index(hgg_data)

    lgg_data = df.loc[df.Grade == "LGG", :]
    lgg_data = reset_df_index(lgg_data)

    # This shuffles the data. Nise.
    hgg_data = hgg_data.loc[np.random.permutation(hgg_data.shape[0]), :]
    lgg_data = lgg_data.loc[np.random.permutation(lgg_data.shape[0]), :]

    # Don't forget to reset the index.
    hgg_data = reset_df_index(hgg_data)
    lgg_data = reset_df_index(lgg_data)

    # Take train:(1-train) split here.
    hgg_train_ix = int(train_split * hgg_data.shape[0])
    lgg_train_ix = int(train_split * lgg_data.shape[0])

    hgg_train = hgg_data.loc[:hgg_train_ix, :]
    hgg_test = hgg_data.loc[hgg_train_ix:, :]

    lgg_train = lgg_data.loc[:lgg_train_ix, :]
    lgg_test = lgg_data.loc[lgg_train_ix:, :]

    train = pd.concat([hgg_train, lgg_train], axis=0)
    test = pd.concat([hgg_test, lgg_test], axis=0)

    train = reset_df_index(train)
    test = reset_df_index(test)

    # Shuffle the data again.
    train = train.loc[np.random.permutation(train.shape[0]), :]
    test = test.loc[np.random.permutation(test.shape[0]), :]

    train = reset_df_index(train)
    test = reset_df_index(test)

    print("Data split has been set-up.")

    return train, test


def prepare_dir_structure(root_dir):
    """
    /data
        |
        | - train
        | - test
    """

    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(os.path.join(root_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "test"), exist_ok=True)
    print("Data directory structure created.")


def copy_data(src_dir, split_type, dst):

    datapoints = split_type["BraTS_2020_subject_ID"].tolist()

    for datapoint in datapoints:
        # id = datapoint.split("_")[-1]
        data_dir = os.path.join(src_dir, datapoint)

        assert len(os.listdir(data_dir)) == 5, "Some files seem to be missing...?"

        for data_file in os.listdir(data_dir):
            try:
                shutil.copy(
                    src=os.path.join(data_dir, data_file), 
                    dst=os.path.join(dst, data_file)
                    )
            except Exception:
                print("Exception in copying data. Oof.")

    print("Copied data from {} to {}.".format(src_dir, dst))


if __name__ == "__main__":
    train_data, test_data = prepare_data("./name_mapping.csv")

    src = "/home/antpc/Documents/data_mount/BRATS2020/MICCAI_BraTS2020_TrainingData"

    print("Train data: {} samples".format(train_data.shape[0]))
    print("Test data: {} samples".format(test_data.shape[0]))

    prepare_dir_structure("/home/antpc/Documents/data_mount/BRATS2020/data")
    
    copy_data(src_dir=src, split_type=train_data, dst="/home/antpc/Documents/data_mount/BRATS2020/data/train")
    copy_data(src_dir=src, split_type=test_data, dst="/home/antpc/Documents/data_mount/BRATS2020/data/test")

    print("Data has been copied and ready for Dataset consumption")
