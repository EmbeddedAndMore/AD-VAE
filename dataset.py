import os
import glob

from PIL import Image
from joblib import Parallel, delayed
from torch.utils.data import Dataset
import torchvision.transforms as T


class TrashDataset(Dataset):

    def __init__(self, root_dir, train=True, transform=None, grayscale = True):
        self.images = []
        self.root_dir = root_dir
        self.transform = transform
        if train:
            self.dataset_path = os.path.join(root_dir,"train/notrash/**/*.png")
        else:
            self.dataset_path = os.path.join(root_dir,"test/**/*.png")
        self.files = glob.glob(self.dataset_path, recursive=True)

        self.files = self.files[:int(len(self.files)/100) * 100]

        if not transform:
            # if no transform is passed do a resize
            transformations = [T.Resize((256,256)), T.ToTensor()]
            if grayscale:
                transformations.insert(1, T.Grayscale())
            tf = T.Compose(transformations)
            self.transform = tf


        results = Parallel(n_jobs=10,verbose=10,batch_size=100)(delayed(self._load_images)(img_file) for img_file in self.files)
        self.images = results

    def _load_images(self, img_path):
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = self.images[idx]

        label = "notrash" if ("/notrash" in img_path) else "trash"
        return img, label