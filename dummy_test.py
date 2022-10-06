import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymatreader import read_mat
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import timeit
from time import process_time
def load_Mat_after_approx(name):
    dataAfterApprox=read_mat(name)
    dataAfterApproxDf = pd.DataFrame(dataAfterApprox['M'])
    return dataAfterApproxDf


def load_Mat_befor_approx(name):
    dataBeforApprox=read_mat(name)
    dataBeforApproxDf = pd.DataFrame(dataBeforApprox['mat'])
    return dataBeforApproxDf


vector=np.random.rand(1280)
dataAfterApproxDf=load_Mat_after_approx(name='HSVD.mat')
print(dataAfterApproxDf)

dataBeforApproxDf=load_Mat_befor_approx(name='weightmat.mat')
#print(dataBeforApproxDf )

start1 = process_time()
vectorMultBevorApp=dataBeforApproxDf.dot(vector)
end1 = process_time()
time1=end1-start1

start2 = process_time()
vectorMultAfterApp=dataAfterApproxDf.dot(vector)
end2 = process_time()
time2=end2-start2

mae=mean_absolute_error(vectorMultAfterApp,vectorMultBevorApp)
mse = mean_squared_error(vectorMultAfterApp,vectorMultBevorApp)
print(vectorMultBevorApp)
print(vectorMultAfterApp)
print('mae=',mae)
print('mse=',mse)
print('time with original MAtrix',time1)
print('time with H- Matrix',time2)

