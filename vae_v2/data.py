import torch
import torchvision.transforms as T

def get_transform(opt):
    """Transforms images"""
    transform = []
    if opt.rotate == True:
        transform.append(T.RandomRotation(0.5))
    transform.append(T.ColorJitter(brightness=opt.brightness))
    transform.append(T.Resize((opt.cropsize, opt.cropsize), interpolation=2))
    transform.append(T.ToTensor())
    if opt.channels == 3:
        transform.append(T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    elif opt.channels == 1:
        transform.append(T.Normalize((0.5), (0.5)))

    return T.Compose(transform)


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt, dataset):
        """Create a dataset instance given the name [dataset_mode] and a multi-threaded data loader."""
        self.opt = opt
        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            num_workers=int(opt.num_threads),
            drop_last=True)

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            # if i * self.opt.batch_size >= self.opt.max_dataset_size:
            #     break
            yield data