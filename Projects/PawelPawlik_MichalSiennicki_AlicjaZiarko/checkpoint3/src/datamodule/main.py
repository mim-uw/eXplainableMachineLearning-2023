import random

import cv2
import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from torchvision.transforms.functional import to_pil_image


class Augmentor(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, item):
        res = self.dataset.__getitem__(item)
        return self.transform(res[0]), res[1]


class MicroscopeAug:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            device = img.device
            img = img.numpy()
            img = img.transpose(1, 2, 0)
            circle = cv2.circle((np.ones(img.shape) * 255).astype(np.uint8),  # image placeholder
                                (img.shape[0] // 2, img.shape[1] // 2),  # center point of circle
                                random.randint(img.shape[1] // 2 - 30, img.shape[1] // 2 - 0),  # radius
                                (0, 0, 0),  # color
                                -1)
            # circle = cv2.blur(circle, tuple([random.randint(0, 50)] * 2))

            img = np.multiply(img, circle - 255)
            img = img.transpose(2, 0, 1)
            img = torch.Tensor(img, device=device)
        return img


class DataModule(LightningDataModule):
    def __init__(self, dataset_train, dataset_val, batch_size: int):
        super().__init__()
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(190),
            transforms.ColorJitter(0.08, 0.08, 0.08, 0.1),
            transforms.RandomPerspective(0.1),
            transforms.RandomErasing(scale=(0.01, 0.1)),
            transforms.RandomRotation(180),
            MicroscopeAug(0.7),
        ])
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(190),
        ])
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(Augmentor(self.dataset_train, self.train_transform), batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(Augmentor(self.dataset_val, self.val_transform), batch_size=self.batch_size, shuffle=True)

    def xai_dataloader(self):
        return DataLoader(Augmentor(self.dataset_train, self.val_transform), batch_size=self.batch_size, shuffle=False)


class CovidDataModule(DataModule):
    def __init__(self, dataset_train, dataset_val, batch_size: int):
        super().__init__(dataset_train, dataset_val, batch_size)
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(0.08, 0.08, 0.08, 0.1),
            transforms.RandomPerspective(0.1),
            transforms.RandomErasing(scale=(0.01, 0.1)),
        ])
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])


@hydra.main(config_path="../conf/datamodule", config_name="covid", version_base=None)
def main(config):
    d = instantiate(config)

    for i, (img, c) in enumerate(d.xai_dataloader()):
        print(i, img.shape, img.dtype, c)
        for j in range(img.shape[0]):
            print(f"class {c[j]}", f"id {j+i*img.shape[0]}")
            cv_image = cv2.cvtColor(np.array(to_pil_image(img[j])), cv2.COLOR_RGB2BGR)
            cv2.imshow("img", cv_image)
            cv2.waitKey(0)


if __name__ == "__main__":
    main()
