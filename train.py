import torch
import torch.nn as nn
import models
from dataloader import Loader
from config import get_config
from parserlist import get_savePathOfModelList

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = get_config()

def getDataLoaderDict(idxModel): # idxModel is tuple
    print("\nStarted Preprocessing")
    trainDataLoaderDict = {}
    testDataLoaderDict = {}

    for idx in idxModel:
        print("-Preprocessing model{} dataloader".format(idx))
        loader = Loader(idx)
        trainLoader, testLoader = loader.getLoader()

        trainDataLoaderDict[idx] = trainLoader
        testDataLoaderDict[idx] = testLoader
        print("-Completed Preprocessing model{} dataloader\n".format(idx))
    print("All Completed Preprocessing\n")
    return trainDataLoaderDict, testDataLoaderDict


def train():
    idxModel = config.idxModel

    trainDataLoaderDict, testDataLoaderDict = getDataLoaderDict(idxModel)
    saveModelPath = get_savePathOfModelList()

    metricDict = {}

    print("Learning Start")
    for idx in idxModel:
        print("\n-Model{} Learning Start".format(idx))

        lr = eval("config.lr{}".format(idx))
        num_epoch = config.num_epoch
        modelPath = saveModelPath[idx-1]


        model = eval("models.Model{}()".format(idx))
        model = model.to(device)

        train_loader = trainDataLoaderDict[idx]
        test_loader = testDataLoaderDict[idx]

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        total_step = len(train_loader)
        for epoch in range(num_epoch):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                labels = labels.float()

                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # if (idx+1) % 100 == 0:
                print("--Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epoch, i + 1,
                                                                             total_step, loss.item()))
        print("-Model {} Learning Complete".format(idx))

        print("-Model{} Start Test".format(idx))
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0

            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)

                if outputs.data >= 0.5:
                    predicted = 1
                else:
                    predicted = 0

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        metricDict[idx] = 100*correct/total
        print("-Model{} Complete Test".format(idx))
        print("-Test Accuracy: {}\n".format(metricDict[idx]))

        model.save(model.state_dict(), modelPath)
        print("[Model{} Save Complete PATH[{}]".format(idx, modelPath))

    print("Learning Complete")

    print("\n\nMETRIC")
    for idx in idxModel:
        print("-Model{} Accuracy: {}".format(idx, metricDict[idx]))



