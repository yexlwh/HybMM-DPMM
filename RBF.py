# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 12:06:37 2017

@author: yexlwh
"""
import tensorflow as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt;
import sklearn as skl
from dataCombinedX1 import dataCombineX
import math as math

def RBF(xs,centerSize,outSize,K,k):
    Weights = tf.Variable(tf.random_normal([centerSize, outSize,K]))
    biases = tf.Variable(tf.zeros([1, outSize,centerSize]) + 0.1)
    
    tempCen=np.arange(centerSize).reshape(1,centerSize)
    #tempCen=np.concatenate((tempCen1,tempCen1),axis=1)
    gaussianCen=tf.to_float(tf.constant(tempCen,shape=(1,centerSize)))
    gaussianKernel=tf.exp(-1*tf.square(xs-gaussianCen))
    predictValue=tf.matmul(gaussianKernel,Weights[:,:,k])+biases[:,:,k]
    return predictValue 