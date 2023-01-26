from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torchvision.datasets import ImageFolder, VisionDataset


class CovidDataset(ImageFolder):
    def __init__(self, train_val: str):
        path = Path(__file__).parent / "covid/Lung Segmentation Data/Lung Segmentation Data" / train_val
        super().__init__(str(path), is_valid_file=lambda p: "masks" not in p)


class MelanomaDataset(VisionDataset):
    def __init__(self, train_val: str):
        super().__init__("")
        self.df = self.all_df().iloc[self.train_ids(train_val)]

    @classmethod
    def path(cls) -> Path:
        return Path(__file__).parent / "melanoma"

    @classmethod
    def all_df(cls):
        return pd.read_csv(cls.path() / "train_concat.csv")

    @classmethod
    def train_ids(cls, train_val: str):
        res = StratifiedKFold(3, shuffle=True, random_state=69).split(cls.all_df(), cls.all_df()["target"])
        return next(res)[0 if train_val == "Train" else 1]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        img = Image.open(f"{self.path()}/train/train/" + row["image_name"] + ".jpg")
        return img, row["target"]

    def __len__(self):
        return len(self.df)
