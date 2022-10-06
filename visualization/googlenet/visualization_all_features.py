import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatreader import read_mat
import matplotlib


HMatrixrank=np.array(read_mat('HMatrixSVDrank.mat')['v'])
MemoryHMatrixSVDrank=np.array(read_mat('MemorySVDrank.mat')['v'])
MultwithMatrixSVDrank=np.array(read_mat('MultwithMatrixSVDrank.mat')['v'])
MultwithHMatrixSVDrank=np.array(read_mat('MultwithHMatriSVDrank.mat')['v'])
timeBuildHMatrixSVDrank=np.array(read_mat('timeBuildHMatrixSVDrank.mat')['v'])
HMatrixrank=np.array(read_mat('HMatrixACArank.mat')['v'])
MemoryHMatrixACArank=np.array(read_mat('MemoryACArank.mat')['v'])
MultwithMatrixACArank=np.array(read_mat('MultwithMatrixACArank.mat')['v'])
MultwithHMatrixACArank=np.array(read_mat('MultwithHMatriACArank.mat')['v'])
timeBuildHMatrixACArank=np.array(read_mat('timeBuildHMatrixACArank.mat')['v'])
HMatrixrank=np.array(read_mat('HMatrixQRrank.mat')['v'])
MemoryHMatrixQRrank=np.array(read_mat('MemoryQRrank.mat')['v'])
MultwithMatrixQRrank=np.array(read_mat('MultwithMatrixQRrank.mat')['v'])
MultwithHMatrixQRrank=np.array(read_mat('MultwithHMatrixQRrank.mat')['v'])
timeBuildHMatrixQRrank=np.array(read_mat('timeBuildHMatrixQRrank.mat')['v'])

timedeltaSVDrank=MultwithMatrixSVDrank-MultwithHMatrixSVDrank;
matplotlib.rc('xtick', labelsize=25) 
matplotlib.rc('ytick', labelsize=25)



fig, axs = plt.subplots(2, 2)
fig.suptitle('Approximation of MobileNet v2, weightmatrix with SVD with rank',fontsize=25,weight="bold")
axs[0, 0].plot(HMatrixrank,timeBuildHMatrixSVDrank,linewidth=7.0)
axs[0, 0].set_title('Time to build H-matrix with SVD in sec',fontsize=20,weight="bold")
axs[0, 0].set_ylabel('Time in sec',fontsize=20,weight="bold")
axs[0, 0].set_xlabel('Rank',fontsize=20,weight="bold")
axs[1, 0].plot(HMatrixrank,MemoryHMatrixSVDrank,linewidth=7.0)
axs[1, 0].set_title('Memory of H-matrix',fontsize=20,weight="bold")
axs[1, 0].set_ylabel('Memory in MB',fontsize=20,weight="bold")
axs[1, 0].set_xlabel('Rank',fontsize=20,weight="bold")
axs[0, 1].plot(HMatrixrank,timedeltaSVDrank,linewidth=7.0)
axs[0, 1].set_title('Time saving matrix-vector multiplication with H-matrix in sec',fontsize=20,weight="bold")
axs[0, 1].set_ylabel('Time in sec',fontsize=20,weight="bold")
axs[0, 1].set_xlabel('Rank',fontsize=20,weight="bold")
axs[1, 1].plot(HMatrixrank,MultwithHMatrixSVDrank, label='with H-matrix',linewidth=7.0)
axs[1, 1].plot(HMatrixrank,MultwithMatrixSVDrank, label='with normal matrix',linewidth=7.0)
axs[1, 1].set_title('Time saving matrix-vector multiplication per 50000 operations ',fontsize=20,weight="bold")
axs[1, 1].set_ylabel('Time in sec',fontsize=20,weight="bold")
axs[1, 1].set_xlabel('Rank',fontsize=20,weight="bold")
axs[1,1].legend(fontsize=20)


timedeltaACArank=MultwithMatrixACArank-MultwithHMatrixACArank;

fig, axs = plt.subplots(2, 2)
fig.suptitle('Approximation of MobileNet v2, weightmatrix with ACA with rank',fontsize=25,weight="bold")
axs[0, 0].plot(HMatrixrank,timeBuildHMatrixACArank,linewidth=7.0)
axs[0, 0].set_title('Time to build H-matrix with ACA in sec',fontsize=20,weight="bold")
axs[0, 0].set_ylabel('Time in sec',fontsize=20,weight="bold")
axs[0, 0].set_xlabel('Rank',fontsize=20,weight="bold")
axs[1, 0].plot(HMatrixrank,MemoryHMatrixACArank,linewidth=7.0)
axs[1, 0].set_title('Memory of H-matrix',fontsize=20,weight="bold")
axs[1, 0].set_ylabel('Memory in MB',fontsize=20,weight="bold")
axs[1, 0].set_xlabel('Rank',fontsize=20,weight="bold")
axs[0, 1].plot(HMatrixrank,timedeltaACArank,linewidth=7.0)
axs[0, 1].set_title('Time saving matrix-vector multiplication with H-matrix in sec',fontsize=20,weight="bold")
axs[0, 1].set_ylabel('Time in sec',fontsize=20,weight="bold")
axs[0, 1].set_xlabel('Rank',fontsize=20,weight="bold")
axs[1, 1].plot(HMatrixrank,MultwithHMatrixACArank, label='with H-matrix',linewidth=7.0)
axs[1, 1].plot(HMatrixrank,MultwithMatrixACArank, label='with normal matrix',linewidth=7.0)
axs[1, 1].set_title('Time saving matrix-vector multiplication per 50000 operations ',fontsize=20,weight="bold")
axs[1, 1].set_ylabel('Time in sec',fontsize=20,weight="bold")
axs[1, 1].set_xlabel('Rank',fontsize=20,weight="bold")
axs[1,1].legend(fontsize=20)


timedeltaQRrank=MultwithMatrixQRrank-MultwithHMatrixQRrank;

fig, axs = plt.subplots(2, 2)
fig.suptitle('Approximation of MobileNet v2, weightmatrix with QR with rank',fontsize=25,weight="bold")
axs[0, 0].plot(HMatrixrank,timeBuildHMatrixQRrank,linewidth=7.0)
axs[0, 0].set_title('Time to build H-matrix with SVD in sec',fontsize=20,weight="bold")
axs[0, 0].set_ylabel('Time in sec',fontsize=20,weight="bold")
axs[0, 0].set_xlabel('Rank',fontsize=20,weight="bold")
axs[1, 0].plot(HMatrixrank,MemoryHMatrixQRrank,linewidth=7.0)
axs[1, 0].set_title('Memory of H-matrix',fontsize=20,weight="bold")
axs[1, 0].set_ylabel('Memory in MB',fontsize=20,weight="bold")
axs[1, 0].set_xlabel('Rank',fontsize=20,weight="bold")
axs[0, 1].plot(HMatrixrank,timedeltaQRrank,linewidth=7.0)
axs[0, 1].set_title('Time saving matrix-vector multiplication with H-matrix in sec',fontsize=20,weight="bold")
axs[0, 1].set_ylabel('Time in sec',fontsize=20,weight="bold")
axs[0, 1].set_xlabel('Rank',fontsize=20,weight="bold")
axs[1, 1].plot(HMatrixrank,MultwithHMatrixQRrank, label='with H-matrix',linewidth=7.0)
axs[1, 1].plot(HMatrixrank,MultwithMatrixQRrank, label='with normal matrix',linewidth=7.0)
axs[1, 1].set_title('Time saving matrix-vector multiplication per 50000 operations ',fontsize=20,weight="bold")
axs[1, 1].set_ylabel('Time in sec',fontsize=20,weight="bold")
axs[1, 1].set_xlabel('Rank',fontsize=20,weight="bold")
axs[1,1].legend(fontsize=20)
plt.show()


