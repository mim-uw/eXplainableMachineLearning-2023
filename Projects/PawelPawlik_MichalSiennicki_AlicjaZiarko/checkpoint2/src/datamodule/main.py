import cv2
import hydra
import numpy as np
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


class DataModule(LightningDataModule):
    def __init__(self, dataset_train, dataset_val, batch_size: int):
        super().__init__()
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            transforms.Normalize(128, 128),
            transforms.RandomPerspective(0.1),
            transforms.RandomErasing(scale=(0.01, 0.1)),
            transforms.RandomRotation(180)
        ])
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(128, 128),
        ])
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(Augmentor(self.dataset_train, self.train_transform), batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(Augmentor(self.dataset_val, self.val_transform), batch_size=self.batch_size, shuffle=True)

    def xai_dataloader(self):
        return DataLoader(Augmentor(self.dataset_train, self.val_transform), batch_size=self.batch_size, shuffle=True)


@hydra.main(config_path="../conf/datamodule", config_name="melanoma", version_base=None)
def main(config):
    print(config)
    d = instantiate(config)

    for i, (img, c) in enumerate(d.train_dataloader()):
        print(i, img.shape, img.dtype, c)
        cv_image = cv2.cvtColor(np.array(to_pil_image(128 * (img[0] + 1))), cv2.COLOR_RGB2BGR)

        cv2.imshow("img", cv_image)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
