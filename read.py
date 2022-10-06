import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymatreader import read_mat


def load_Mat_after_approx(name):
    dataAfterApprox=read_mat(name)
    dataAfterApproxDf = pd.DataFrame(dataAfterApprox['M'])
    return dataAfterApproxDf


def load_Mat_befor_approx(name):
    dataBeforApprox=read_mat(name)
    dataBeforApproxDf = pd.DataFrame(dataBeforApprox['mat'])
    return dataBeforApproxDf



dataAfterApproxDf=load_Mat_after_approx(name='HSVD.mat')
dataAfterApproxDf.hist()

dataBeforApproxDf=load_Mat_befor_approx(name='weightmat.mat')
print(dataBeforApproxDf )


