import argparse

parser = argparse.ArgumentParser()

for i in range(1, 26):
    parser.add_argument('--dataroot{}'.format(i), type=str, default='./dataset/model{}'.format(i), help='dataset path of model{}'.format(i))
    parser.add_argument('--savePathOfModel{}'.format(i), type=str, default='./models/model{}.ckpt'.format(i))

defaultImageSizeList = [(870, 100), (,), (,), (,), (,),
                        (,), (,), (,), (,), (,),
                        (,), (,), (,), (,), (,),
                        (,), (,), (,), (,), (,),
                        (,), (,), (,), (,), (,),] # 25 tuple
for i, imgSize in enumerate(defaultImageSizeList):
    parser.add_argument('--image_size{}'.format(i+1), type=tuple, default=imgSize, help='image size of model{}'.format(i+1))

parser.add_argument('--batch_size', type=int, default=16, help='minibatch size')

parser.add_argument('--num_workers', type=int, default=1, help='number of workers')

parser.add_argument('--num_epoch', type=int, default=5, help='number of epochs')

defaultLearningRateList = [0.001, 0.001, 0.001, 0.001, 0.001,
                           0.001, 0.001, 0.001, 0.001, 0.001,
                           0.001, 0.001, 0.001, 0.001, 0.001,
                           0.001, 0.001, 0.001, 0.001, 0.001,
                           0.001, 0.001, 0.001, 0.001, 0.001] # 25 float
for i, lr in enumerate(defaultLearningRateList):
    parser.add_argument('--lr{}'.format(i+1), type=float, default=lr, help='learning rate of model{}'.format(i+1))

parser.add_argument('--idxModel', type=tuple, default=(i for i in range(1, 26)), help='1~25 model num. dafult is all model')


def get_config():
    config = parser.parse_args()
    return config
