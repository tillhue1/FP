
from extract_deep_nn_weight_matrix import get_mobilenet_target_mats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
target_mats = get_mobilenet_target_mats()
target_mats = target_mats.pop()
target_mats_array = np.array(target_mats)
#print(target_mats_array)
target_mats_array=np.array(target_mats_array)
target_mats_newarr = target_mats_array.reshape(1*1000,1280).astype(np.float64)
#target_mats_newarr=np.random.rand(10,16).astype(np.float64)
#print(target_mats_newarr.shape)
#print(target_mats_newarr.dtype)
#print(target_mats_newarr)
target_mats_dataframe=pd.DataFrame(target_mats_newarr)
print(target_mats_dataframe)
#print(target_mats_newarr[7,1003])
#sio.savemat('weightmat_test1016.mat',{'mat':target_mats_newarr})
#print(target_mats_dataframe)

#.to_csv("weightmat.csv")
