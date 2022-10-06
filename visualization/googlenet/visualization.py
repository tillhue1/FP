import matplotlib.pyplot as plt
import matplotlib 
import numpy as np
import pandas as pd
from pymatreader import read_mat

HMatrixrank=np.array(read_mat('HMatrixSVDrank.mat')['v'])
MultwithHMatrixSVDrank=np.array(read_mat('MultwithHMatriSVDrank.mat')['v'])
MultwithHMatrixACArank=np.array(read_mat('MultwithHMatriACArank.mat')['v'])
MultwithHMatrixQRrank=np.array(read_mat('MultwithHMatrixQRrank.mat')['v'])
MultwithMatrixACArank=np.array(read_mat('MultwithMatrixQRrank.mat')['v'])

matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)
print(MultwithHMatrixSVDrank)
print(np.mean(MultwithMatrixACArank))

plt.plot(HMatrixrank,MultwithHMatrixACArank,linewidth=3.0,label='ACA')
plt.plot(HMatrixrank,MultwithHMatrixQRrank,linewidth=3.0,label='QR')
plt.plot(HMatrixrank,MultwithHMatrixSVDrank,linewidth=3.0,label='SVD')
plt.plot(HMatrixrank,MultwithMatrixACArank,linewidth=3.0,label='without approximation')
plt.xlabel('Rank',fontsize=18,weight="bold")
plt.ylabel('time in sec',fontsize=18,weight="bold")
plt.title('Time 100000 matrix-vecotr multiplication with different approximation methods using MobileNet v2',fontsize=20,weight="bold")
plt.legend(fontsize=20)
plt.show()
plt.savefig('Time.png')
