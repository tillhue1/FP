import matplotlib.pyplot as plt
import matplotlib 
import numpy as np
import pandas as pd
from pymatreader import read_mat

HMatrixrank=np.array(read_mat('HMatrixSVDrank.mat')['v'])
MultwithHMatrixSVDrank=np.array(read_mat('MultwithHMatrixSVDrank.mat')['v'])
MultwithHMatrixQRrank=np.array(read_mat('MultwithHMatrixQRrank.mat')['v'])
MultwithMatrixACArank=np.array(read_mat('MultwithMatrixQRrank.mat')['v'])
MultwithHMatrixSVDrankauto=np.array(read_mat('MultwithHMatriSVDrankauto.mat')['v'])
MultwithHMatrixQRrankauto=np.array(read_mat('MultwithHMatrixQRrankauto.mat')['v'])
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)


plt.plot(HMatrixrank,MultwithHMatrixQRrank,linewidth=3.0,label='QR TCardBSPPartStrat')
plt.plot(HMatrixrank,MultwithHMatrixSVDrank,linewidth=3.0,label='SVD TCardBSPPartStrat')
plt.plot(HMatrixrank,MultwithHMatrixSVDrankauto,linewidth=3.0,label='SVD TAutoBSPPartStrat')
plt.plot(HMatrixrank,MultwithHMatrixQRrankauto,linewidth=3.0,label='QR TAutoBSPPartStrat')
plt.plot(HMatrixrank,MultwithMatrixACArank,linewidth=3.0,label='without approximation')
plt.xlabel('Rank',fontsize=18,weight="bold")
plt.ylabel('time in sec',fontsize=18,weight="bold")
plt.title('Time 100000 matrix-vecotr multiplication with different approximation methods and cluster methods using MobileNet v2',fontsize=20,weight="bold")
plt.legend(fontsize=20)
plt.show()

