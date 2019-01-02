from config import get_config

config = get_config()

def get_datarootList():
    datarootList = [eval("config.dataroot{}".format(i)) for i in range(1, 26)]
    return datarootList


def get_savePathOfModelList():
    savePathOfModelList = [eval("config.savePathOfModel{}".format(i)) for i in range(1, 26)]
    return savePathOfModelList


def get_imageSizeList():
    imageSizeList = [eval("config.image_size{}".format(i)) for i in range(1, 26)]
    return imageSizeList


def get_learningRateList():
    lrList = [eval("config.lr{}".format(i)) for i in range(1, 26)]
    return lrList