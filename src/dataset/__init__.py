import os
import random
from glob import glob

import torch
import cv2

import pandas as pd
from torch.utils.data.dataset import Dataset
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


from .transformations import *

seed = 40302
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)


def get_split(data, size=0.2):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=size, random_state=seed)

    for a, b in sss.split(np.arange(data.shape[0]), data.label):
        train_idx, test_idx = a, b

    return train_idx, test_idx


def load_data():

    # load datasets from csv and folders
    # 1. DFDC
    dfdc_folders = glob(os.getcwd() + "\\dfdc\\dfdc*")
    dfdc_folders = sorted(dfdc_folders, key=lambda x: x)
    all_dataframes = []
    for train_dir in dfdc_folders:
        df = pd.read_csv(os.path.join(train_dir, "metadata.csv"))
        df["path"] = df["filename"].apply(lambda x: os.path.join(train_dir, x.split(".")[0]))
        all_dataframes.append(df)

    dfdc_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
    dfdc_df["origin"] = "dfdc"
    dfdc_df.drop(["split", "original"], axis=1, inplace=True)

    # 2. CelebDF
    celeb_df = pd.read_csv("./celeb_metadata.csv")
    celeb_df["path"] = celeb_df["Unnamed: 0"].apply(
        lambda x: os.path.join("h:\\project\\CelebDF-v2\\faces", x.split(".")[0])
    )
    celeb_df.drop(["img_dir", "fullpath"], axis=1, inplace=True)
    celeb_df.rename(columns={"Unnamed: 0": "filename"}, inplace=True)
    celeb_df = celeb_df[["filename", "label", "path", "origin"]]
    df2 = pd.read_csv("./CelebDF-v2/List_of_testing_videos.txt", delimiter=" ", header=None)
    df2[1] = df2[1].apply(lambda s: s.split("/")[-1])
    test_celeb = celeb_df.loc[celeb_df.filename.isin(df2[1].tolist())].reset_index(drop=True)
    celeb_df.drop(test_celeb.index, inplace=True)
    celeb_df.reset_index(drop=True, inplace=True)

    # 3 . Youtube Faces Extracted
    ytf_df = pd.read_csv("./youtube_faces_dataset/youtube_faces_50f.csv")
    ytf_df["label"] = "REAL"
    ytf_df["origin"] = "ytf"
    ytf_df["path"] = ytf_df["videoID"].apply(
        lambda x: os.path.join("./youtube_faces_dataset/faces", x)
    )
    ytf_df.head()

    ytf_df.drop(
        [
            "Unnamed: 0",
            "personName",
            "imageHeight",
            "imageWidth",
            "videoDuration",
            "averageFaceSize",
            "numVideosForPerson",
        ],
        axis=1,
        inplace=True,
    )
    ytf_df.rename(columns={"videoID": "filename"}, inplace=True)
    ytf_df = ytf_df[["filename", "label", "path", "origin"]]
    x = [False for n in range(len(ytf_df.path))]
    for i, path in enumerate(ytf_df.path.tolist()):
        already_present_count = len(glob(path + "/*"))
        if already_present_count >= 50:
            x[i] = True
    ytf_df = ytf_df[x].reset_index(drop=True)

    # Separate test dataset before duplicate real frames from celebdf and dfdc

    df = pd.concat([dfdc_df, celeb_df, ytf_df], ignore_index=True, sort=False)
    df["frames"] = 1
    df["inv_frame"] = 0

    _, _test = get_split(df, 0.2)
    test_df = df.loc[_test, :]
    test_df.reset_index(drop=True, inplace=True)

    df.drop(_test, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # get validation set before duplications.
    train_idx, val_idx = get_split(df, 0.2)
    train_df = df.loc[train_idx, :].reset_index(drop=True)
    valid_df = df.loc[val_idx, :].reset_index(drop=True)

    # duplicate data labeled real in Dfdc and Celebdf
    real_dfdc = train_df[
        (train_df["label"] == "REAL") & (train_df["origin"] == "dfdc")
    ].reset_index(drop=True)

    real_celeb = train_df[
        (train_df["label"] == "REAL") & (train_df["origin"] == "CelebDF-v2")
    ].reset_index(drop=True)

    for i in range(2, 4):
        if i <= 3:
            real_celeb.frames = i
            real_celeb_i = real_celeb.copy()
            real_celeb_i.inv_frame = 1
            train_df = pd.concat([train_df, real_celeb_i], ignore_index=True, sort=False)
        real_dfdc.frames = i
        real_dfdc_i = real_dfdc.copy()
        real_dfdc_i.inv_frame = 1
        train_df = pd.concat([train_df, real_dfdc_i], ignore_index=True, sort=False)

    # undersample fakes to have around 50-50% of training samples
    train_df.drop(
        train_df[(train_df["label"] == "FAKE") & (train_df["origin"] == "dfdc")].sample(4000).index,
        inplace=True,
    )

    train_df.drop(
        train_df[(train_df["label"] == "FAKE") & (train_df["origin"] == "CelebDF-v2")]
        .sample(1300)
        .index,
        inplace=True,
    )
    train_df.reset_index(drop=True, inplace=True)

    valid_df.drop(
        valid_df[(valid_df["label"] == "FAKE") & (valid_df["origin"] == "dfdc")].sample(1000).index,
        inplace=True,
    )

    valid_df.drop(
        valid_df[(valid_df["label"] == "FAKE") & (valid_df["origin"] == "CelebDF-v2")]
        .sample(400)
        .index,
        inplace=True,
    )
    valid_df.reset_index(drop=True, inplace=True)

    return train_df, valid_df, test_df, test_celeb


class VideoDataset(Dataset):
    def __init__(self, df, frames_to_use=50, im_size=112, transform=None):
        assert frames_to_use <= 50, "please use max 50 frames"

        self.data = df.reset_index(drop=True).copy()
        self.transform = transform
        self.count = frames_to_use
        self.labels = self.data["label"].values
        self.transforms_inv = transforms_inv(im_size)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        path = self.data["path"][idx] + "/*.png"
        img_list = glob(path)
        frames = []

        # setup the correct starting frame according with the frame block set in the dataframe
        frame_idx = self.data["frames"][idx]

        start_frame = (frame_idx - 1) * self.count
        if len(img_list[start_frame:]) < self.count:
            start_frame = len(img_list[start_frame:]) - 1 - self.count
        end_frame = start_frame + self.count - 1

        # get label
        label = self.labels[idx]

        if label == "FAKE":
            label = 0
        if label == "REAL":
            label = 1

        # put frames of the folder in a list
        # for i,img_path in enumerate(img_list):
        for i in range(start_frame, end_frame + 1):
            image = cv2.imread(img_list[i], cv2.IMREAD_COLOR)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.data["inv_frame"][idx]:
                frames.append(self.transforms_inv(image))
            else:
                frames.append(self.transform(image))
            if len(frames) == self.count:
                break
        frames = torch.stack(frames)

        return frames, label
