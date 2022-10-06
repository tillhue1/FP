import torch
import torchvision.models as models
import torch
import numpy as np
import pandas as pd
import scipy.io as sio
model = models.googlenet(pretrained=True)
weightmatrix=model.fc.weight.data
print(weightmatrix.shape)
print(weightmatrix)
weightmatrix_array = np.array(weightmatrix)
weightmatrix_newarr = weightmatrix_array.reshape(1000,1024).astype(np.float64)
#sio.savemat('weightmat_googlenet_1000_1024.mat',{'mat':weightmatrix_newarr})
