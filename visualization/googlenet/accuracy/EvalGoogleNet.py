import torch
import torchvision
import torchvision.models as models
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
from pymatreader import read_mat
from time import process_time
from numpy import save, load

def load_Mat_after_approx(name):
    dataAfterApprox=read_mat(name)
    dataAfterApproxDf = pd.DataFrame(dataAfterApprox['M'])
    return dataAfterApproxDf


def set_new_layer(dataAfterApproxDf):
    new_layer_tensor = torch.nn.Parameter(torch.tensor(dataAfterApproxDf.values,dtype=torch.float))
   
    model.fc.weight.data=new_layer_tensor
    return model


def eval(model):
    data_directory = '/home/till/Desktop/Forschungspraxis/Code/C_Code/hlibpro-2.9.1-Ubuntu18.04/hlibpro-2.9.1'
    preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    dataset = \
            datasets.ImageNet(root=data_directory,
                             split='val',
                              transform=preprocess)
    train_set_size = int(len(dataset) * 0.6)
    valid_set_size = len(dataset) - train_set_size
    train, valid = torch.utils.data.random_split(dataset,[train_set_size, valid_set_size])
    validloader = torch.utils.data.DataLoader(valid,batch_size=32)
    model.eval()  #for better acurracy
    match_list = []
    timePerPrediction=[]
    numberIterations=0
    
    for data, labels in validloader:
        numberIterations=numberIterations+1
        start = process_time()
        prediction = model(data)
        end = process_time()
        timePerPrediction.append(end-start)
        _, predictions = torch.max(prediction, 1)
 
        matches = (labels == predictions).tolist()
        match_list.extend(matches)
        if len(match_list) > 100000: #100000       
            break
            accuracy=sum(match_list)/len(match_list)
            timePerIteration=np.average(timePerPrediction)
    timePerIteration=np.average(timePerPrediction)
    accuracy=sum(match_list)/len(match_list)
    return accuracy, numberIterations,timePerIteration


accuracy=[]
floataccuracy= np.array(accuracy, dtype = np.float32)
model = models.googlenet(pretrained=True)
accuracy,numberIterations,timePerIteration=eval(model)
floataccuracy = np.append(floataccuracy, accuracy)
print("Accuracy of pretrained model",accuracy)
print("Time of predicting 1 batch size befor approximation",timePerIteration)
for i in range(0,13):#13
    rank=i*10
    if rank==0:
        rank=1
    str1='HmobileACA'
    strrank=str(rank)
    str3='.mat'
    filename=str1+strrank+str3
    print(filename)
    dataAfterApproxDf=load_Mat_after_approx(name=filename)
    model_new=set_new_layer(dataAfterApproxDf)
    accuracyNew,numberIterationsNew,timePerIterationNew=eval(model)
    print("Accuracy of model with H-Matrix ",accuracyNew, )
    floataccuracy = np.append(floataccuracy, accuracyNew)
    print("Time of predicting 1 batch size after approximation ",timePerIterationNew)
print(floataccuracy)
str4='accucuracy.npy'
filename2=str1+str4
save(filename2, floataccuracy)
data = load(filename2)
print(data)
