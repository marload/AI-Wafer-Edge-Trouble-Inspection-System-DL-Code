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
        self.numModel = numModel - 1 # index operation
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.dataroot = PL.get_datarootList()[self.numModel]
        self.imageSize = PL.get_imageSizeList()[self.numModel]
        self.test_transform = None
        self.train_transform = None

    def getLoader(self):
        print("\nPreprocessing data...")
        print(type(self.numModel))
        train_path = os.path.join(self.dataroot, 'train')
        test_path = os.path.join(self.dataroot, 'test')

        self.get_TrainTransform()
        self.get_TestTransform()

        train_set = ImageFolder(train_path, transform=self.train_transform)
        test_set = ImageFolder(test_path, transform=self.test_transform)
        print(self.imageSize)
        print(type(self.imageSize))
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

    def get_TestTransform(self):
        transform = T.Compose([
            T.Resize(self.imageSize),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5))
        ])
        self.test_transform = transform
        return

    def get_TrainTransform(self):
        if self.numModel == 1:
            transform = T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))
            ])
            self.train_transform = transform
            return

        elif self.numModel == 2:
            transform = T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))
            ])
            self.train_transform = transform
            return

        elif self.numModel == 3:
            transform = T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))
            ])
            self.train_transform = transform
            return

        elif self.numModel == 4:
            transform = T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))
            ])
            self.train_transform = transform
            return

        elif self.numModel == 5:
            transform = T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))
            ])
            self.train_transform = transform
            return

        elif self.numModel == 6:
            transform = T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))
            ])
            self.train_transform = transform
            return

        elif self.numModel == 7:
            transform = T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))
            ])
            self.train_transform = transform
            return

        elif self.numModel == 8:
            transform = T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))
            ])
            self.train_transform = transform
            return

        elif self.numModel == 9:
            transform = T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))
            ])
            self.train_transform = transform
            return

        elif self.numModel == 10:
            transform = T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))
            ])
            self.train_transform = transform
            return

        elif self.numModel == 11:
            transform = T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))
            ])
            self.train_transform = transform
            return
            return transform

        elif self.numModel == 12:
            transform = T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))
            ])
            self.train_transform = transform
            return

        elif self.numModel == 13:
            transform = T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))
            ])
            self.train_transform = transform
            return

        elif self.numModel == 14:
            transform = T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))
            ])
            self.train_transform = transform
            return

        elif self.numModel == 15:
            transform = T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))
            ])
            self.train_transform = transform
            return

        elif self.numModel == 16:
            transform = T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))
            ])
            self.train_transform = transform
            return

        elif self.numModel == 17:
            transform = T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))
            ])
            self.train_transform = transform
            return

        elif self.numModel == 18:
            transform = T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))
            ])
            self.train_transform = transform
            return

        elif self.numModel == 19:
            transform = T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))
            ])
            self.train_transform = transform
            return

        elif self.numModel == 20:
            transform = T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))
            ])
            self.train_transform = transform
            return

        elif self.numModel == 21:
            transform = T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))
            ])
            self.train_transform = transform
            return

        elif self.numModel == 22:
            transform = T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))
            ])
            self.train_transform = transform
            return

        elif self.numModel == 23:
            transform = T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))
            ])
            self.train_transform = transform
            return

        elif self.numModel == 24:
            transform = T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))
            ])
            self.train_transform = transform
            return

        elif self.numModel == 25:
            transform = T.Compose([
                T.Resize(self.imageSize),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))
            ])
            self.train_transform = transform
            return


