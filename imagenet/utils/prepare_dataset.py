import torch
import random
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
import os

def prepare_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    val_corrupt_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return train_transform, val_transform, val_corrupt_transform

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_dataloader(dataset, args, shuffle=False, drop_last=False):
    return torch.utils.data.DataLoader(dataset, 
                                batch_size=args.batch_size, 
                                shuffle=shuffle, 
                                num_workers=args.workers, 
                                worker_init_fn=seed_worker, 
                                pin_memory=True, 
                                drop_last=drop_last)


class ImageNetCorruption(ImageNet):
    def __init__(self, root, corruption_name="gaussian_noise", transform=None, is_carry_index=False):
        super().__init__(root, 'val', transform=transform)
        self.root = root
        self.corruption_name = corruption_name
        self.transform = transform
        self.is_carry_index = is_carry_index
        self.load_data()
    
    def load_data(self):
        self.data = torch.load(os.path.join(self.root, 'corruption', self.corruption_name + '.pth')).numpy()
        self.target = [i[1] for i in self.imgs]
        return
    
    def __getitem__(self, index):
        img = self.data[index, :, :, :]
        target = self.target[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.is_carry_index:
            img = [img, index]
        return img, target
    
    def __len__(self):
        return self.data.shape[0]

class ImageNet_(ImageNet):
    def __init__(self, *args, is_carry_index=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_carry_index = is_carry_index
    
    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        if self.is_carry_index:
            if type(img) == list:
                img.append(index)
            else:
                img = [img, index]
        return img, target


