"""
Author: Felix Naser
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import torch
import torchvision.datasets as datasets
import torch.utils.data as data
from PIL import Image
from utils.mypath import MyPath
from torchvision import transforms as tf
from glob import glob

class SewerNet(data.Dataset):
    def __init__(self, root=MyPath.db_root_dir('sewer'), split='Training', 
                    transform=None):
        super(SewerNet, self).__init__()

        self.root = os.path.join(root, split)
        self.transform = transform

        subdirs = []
        for name in os.listdir(self.root):
            subdirs.append(name)

        print(split)
        print(self.root)
        print(subdirs)

        imgs = []
        for i, subdir in enumerate(subdirs):
            subdir_path = os.path.join(self.root, subdir)
            files = sorted(glob(os.path.join(self.root, subdir, '*.jpg')))
            for f in files:
                imgs.append((f, i)) 
        self.imgs = imgs 
        self.classes = subdirs

        self.resize = tf.Resize(128) #256

    def get_image(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.resize(img) 
        return img

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        im_size = img.size
        img = self.resize(img) 
        class_name = self.classes[target]

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {'im_size': im_size, 'index': index, 'class_name': class_name}}

        return out
