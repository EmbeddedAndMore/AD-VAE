import os
import glob

from PIL import Image
from joblib import Parallel, delayed
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import cv2 as cv


class TrashDataset(Dataset):

    def __init__(self, root_dir, train=True, transform=None, grayscale = True, subset=False):
        self.images = []
        self.root_dir = root_dir
        self.transform = transform
        if train:
            self.dataset_path = os.path.join(root_dir,"train/notrash/**/*.png")
        else:
            self.dataset_path = os.path.join(root_dir,"test/**/*.png")
        self.files = glob.glob(self.dataset_path, recursive=True)

        self.files = self.files[:int(len(self.files)/100) * 100]

        if subset:
            if train:
                self.files = np.random.choice(self.files, 15000, replace=False).tolist()
            else:
                self.files = np.random.choice(self.files, 1200, replace=False).tolist()

        if not transform:
            # if no transform is passed do a resize
            transformations = [T.Resize((256,256)), T.ToTensor()]
            if grayscale:
                transformations.insert(1, T.Grayscale())
            tf = T.Compose(transformations)
            self.transform = tf


        results = Parallel(n_jobs=8,verbose=10,batch_size=200, backend="threading")(delayed(self._load_images)(img_file) for img_file in self.files)
        self.images = results

    def _load_images(self, img_path):
        # img = cv.imread(img_path)
        # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = Image.open(img_path).convert('RGB')
        return img

    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
            # img = self.transform(image=img)["image"]

        label = "notrash" if ("/notrash" in img_path) else "trash"
        return {"img": img, "label": label, "path":img_path}