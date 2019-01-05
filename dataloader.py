from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as T

import parserlist as PL

import os

from config import get_config

config = get_config()
batch_size = config.batch_size
num_workers = config.num_workers


class Loader:
    def __init__(self, numModel): # numModel is an integer value between 1 and 25
        self.numModel = numModel# index operation
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.dataroot = PL.get_datarootList()[self.numModel-1]
        self.imageSize = PL.get_imageSizeList()[self.numModel-1]

    def getLoader(self):
        print(self.numModel)
        print("\nPreprocessing data...")
        train_path = os.path.join(self.dataroot, 'train')
        test_path = os.path.join(self.dataroot, 'test')

        test_set = ImageFolder(test_path, transform=T.Compose([
            T.Resize(self.imageSize),
            T.Grayscale(1),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5))
        ]))

        if self.numModel == 1:
            train_set = ImageFolder(train_path, transform=T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5))
            ]))

        elif self.numModel == 2:
            train_set = ImageFolder(train_path, transform=T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5))
            ]))

        elif self.numModel == 3:
            train_set = ImageFolder(train_path, transform=T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5))
            ]))

        elif self.numModel == 4:
            train_set = ImageFolder(train_path, transform=T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5))
            ]))

        elif self.numModel == 5:
            train_set = ImageFolder(train_path, transform=T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5))
            ]))

        elif self.numModel == 6:
            train_set = ImageFolder(train_path, transform=T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5))
            ]))

        elif self.numModel == 7:
            train_set = ImageFolder(train_path, transform=T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5))
            ]))

        elif self.numModel == 8:
            train_set = ImageFolder(train_path, transform=T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5))
            ]))

        elif self.numModel == 9:
            train_set = ImageFolder(train_path, transform=T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5))
            ]))

        elif self.numModel == 10:
            train_set = ImageFolder(train_path, transform=T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5))
            ]))

        elif self.numModel == 11:
            train_set = ImageFolder(train_path, transform=T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5))
            ]))

        elif self.numModel == 12:
            train_set = ImageFolder(train_path, transform=T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5))
            ]))

        elif self.numModel == 13:
            train_set = ImageFolder(train_path, transform=T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5))
            ]))

        elif self.numModel == 14:
            train_set = ImageFolder(train_path, transform=T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5))
            ]))

        elif self.numModel == 15:
            train_set = ImageFolder(train_path, transform=T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5))
            ]))

        elif self.numModel == 16:
            train_set = ImageFolder(train_path, transform=T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5))
            ]))

        elif self.numModel == 17:
            train_set = ImageFolder(train_path, transform=T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5))
            ]))

        elif self.numModel == 18:
            train_set = ImageFolder(train_path, transform=T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5))
            ]))

        elif self.numModel == 19:
            train_set = ImageFolder(train_path, transform=T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5))
            ]))

        elif self.numModel == 20:
            train_set = ImageFolder(train_path, transform=T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5))
            ]))

        elif self.numModel == 21:
            train_set = ImageFolder(train_path, transform=T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5))
            ]))

        elif self.numModel == 22:
            train_set = ImageFolder(train_path, transform=T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5))
            ]))

        elif self.numModel == 23:
            train_set = ImageFolder(train_path, transform=T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5))
            ]))

        elif self.numModel == 24:
            train_set = ImageFolder(train_path, transform=T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5))
            ]))

        elif self.numModel == 25:
            train_set = ImageFolder(train_path, transform=T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5))
            ]))

        train_loader = DataLoader(train_set,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers)
        test_loader = DataLoader(test_set,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=self.num_workers)

        print("Completed Preprocessing!\n")
        return train_loader, test_loader