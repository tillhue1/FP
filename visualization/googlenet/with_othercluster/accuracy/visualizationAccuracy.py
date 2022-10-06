import numpy as np
import pandas as pd
from numpy import save, load
import matplotlib 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from pymatreader import read_mat

dataQR = load('HmobileQRaccucuracy.npy')
dataSVD = load('HmobileSVDaccucuracy.npy')

HMatrixrank=np.array(read_mat('HMatrixSVDrank.mat')['v'])
figure(figsize=(30, 20), dpi=80)
matplotlib.rc('xtick', labelsize=25) 
matplotlib.rc('ytick', labelsize=25) 
AccwithoutApprox=np.ones(dataQR.size)*dataSVD[0]
AccwithoutApprox=np.delete(AccwithoutApprox,0)
dataQR=np.delete(dataQR,0)
dataSVD=np.delete(dataSVD,0)

print(AccwithoutApprox)
print(dataSVD)

plt.plot(HMatrixrank,dataQR,linewidth=8.0,label='QR')
plt.plot(HMatrixrank,dataSVD,linewidth=8.0,label='SVD')
plt.plot(HMatrixrank,AccwithoutApprox,linewidth=8.0,label='without approximation')
plt.xlabel('Rank',fontsize=25,weight="bold")
plt.ylabel('Accuracy',fontsize=25,weight="bold")
plt.title('Accuracy with different approximation methods using GoogleLeNet ',fontsize=30,weight="bold")
plt.legend(fontsize=25)
plt.show()
plt.savefig('Accuracy.png')
