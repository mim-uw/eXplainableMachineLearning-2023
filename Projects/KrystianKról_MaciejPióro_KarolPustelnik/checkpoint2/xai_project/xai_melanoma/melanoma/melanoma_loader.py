import torch.utils.data as data
import numpy as np
import pandas as pd
from PIL import Image

class Melanoma_loader(data.Dataset):
    def __init__(self, root, ann_path, transform=None, target_transform=None):

        self.data_path = root
        self.ann_path = ann_path
        self.transform = transform
        self.target_transform = target_transform
        self.database = pd.read_csv(self.ann_path)

    def _load_image(self, path):
        try:
            im = Image.open(path)
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        idb = self.database.iloc[index]
        filename = idb[0]
        Class = int(idb[5])
        images = self._load_image(self.data_path + '/' + str(filename) + '.jpg')
        if self.transform is not None:
            images = self.transform(images)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return images, Class
	
    def lookup_path(self, index):
        idb = self.database.iloc[index]
        return idb[0]
    
    
    def __len__(self):
        return len(self.database)
