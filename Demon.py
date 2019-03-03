# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 19:58:24 2017

@author: yexlwh
"""
import tensorflow as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt;
import sklearn as skl
from dataCombinedX1 import dataCombineX
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics

from vdpmm_expectationCNN import *
from vdpmm_maximizeCNN import *
from vdpmm_maximizePlusGaussian import *
from vdpmm_expectationPlusGaussian import *
from dp_init import *
import math as math
from RBF import *
from sklearn import cluster, datasets
import tensorflow.contrib.slim as slim

def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs



noisy_circles = datasets.make_circles(n_samples=500, factor=.5,noise=.05)
# #noisy_circles= datasets.make_moons(n_samples=500, noise=.05)
X = noisy_circles[0]
y=noisy_circles[1].reshape(1,500)
dataTemp=X
zTemp=y

#dataInput=sio.loadmat('dataTest1.mat')
#dataTemp=dataInput['data']
#zTemp=dataInput['inputK']
Nz,Dz=dataTemp.shape
Dz=Nz



#Dz=Nz
data=dataTemp[0:Dz,:]
z=zTemp.reshape(1,Dz)

# for i in range(Dz-1):
#    if z[i]==3:
#        data[i,1]=data[i,1]+300
#
# data1=np.zeros((1,2))
# count=0
# for i in range(Dz-1):
#    if z[i]==1:
#        data1[0,1]=data[i,1]+300
#        data1[0,0]=data[i,0]+300
#        count=count+1
#        break
#
# temp=np.zeros((1,2))
# for i in range(Dz-1):
#    if z[i]==1:
#        temp[0,1]=data[i,1]+300
#        temp[0,0]=data[i,0]+300
#        data1=np.r_[data1,temp]
#        count=count+1
#
# Dz=Dz+count
# data=np.r_[data,data1]
data=data/(np.max(data)-np.min(data))


x1,center=dataCombineX(data,50)
x1.shape=(Dz,1)

#vdpmm
K=2
infinite=1
verbose=1
maxits=200
minits=10
eps=0.01
numits = 2;
score = -np.inf;
score_change = np.inf;
cho=0;

centerSize=30
outSize=2
scale=1;
learnRate1=1e-10
learnRate2=1e-1
lambdaPos=0.8
print('lambdaPos:',lambdaPos)

hiden_size=1
xs = tf.placeholder(tf.float32, [None, outSize])
ys = tf.placeholder(tf.float32, [None, outSize])
ys = tf.placeholder(tf.float32, [None, outSize])

l1 = add_layer(xs, outSize, 5, activation_function=tf.nn.sigmoid)
l2 = add_layer(l1, 5, 3, activation_function=tf.nn.sigmoid)
#l3 = add_layer(l2, 6, 5, activation_function=tf.nn.relu)
prediction = add_layer(l2,3, hiden_size*2)#,activation_function=tf.nn.sigmoid)

l21 = add_layer(xs, outSize, 5, activation_function=tf.nn.sigmoid)
l22 = add_layer(l21, 5, 3, activation_function=tf.nn.sigmoid)
#l3 = add_layer(l2, 6, 5, activation_function=tf.nn.relu)
prediction2 = add_layer(l22,3, hiden_size*2)#,activation_function=tf.nn.sigmoid)

# l1 = add_layer(xs, outSize, 3, activation_function=tf.nn.relu)
# l2 = add_layer(l1, 3, 4, activation_function=tf.nn.relu)
# l3 = add_layer(l2, 4, 3, activation_function=tf.nn.relu)
# prediction = add_layer(l3,3, hiden_size*2)#,activation_function=tf.nn.sigmoid)
mean = prediction[:, :hiden_size]+x1
stddev = prediction[:, hiden_size:]*prediction[:, hiden_size:]

mean2 = prediction2[:, :hiden_size]+x1
stddev2 = prediction2[:, hiden_size:]*prediction2[:, hiden_size:]

#loss function for posterior probability network
D=hiden_size
BPlace=tf.placeholder(tf.float32, [D, D, K])
aPlace=tf.placeholder(tf.float32, [K,1])
meanPlace=tf.placeholder(tf.float32, [K, D])
gammasPlace=tf.placeholder(tf.float32, [Dz, K])
lossPos=tf.constant(0.0)
#for k in range(K):
#    tempU=meanPlace[k,:]#tf.to_float(tf.constant(uk[k,:]))
#    loss1=(tempU-mean)
#    tempB=BPlace[:,:,k]#tf.to_float(tf.constant(wk[:,:,k]))
#    tempPro=tf.matmul(loss1,tempB)
#    tempA=aPlace[k]#tf.to_float(tf.constant(params['a'][k]))
#    loss=tempA*tf.reduce_sum(tempPro*loss1,1,keep_dims=True)+tempA*tempB*stddev-0.5*tf.log(stddev)
#    lossSum=tf.reduce_sum(loss)
#
#    lossPos=lossPos+lossSum

sampleNum=Dz
epsilon = tf.random_normal([sampleNum, hiden_size])
#
input_sample = mean + epsilon * stddev

input_sample2 = mean2 + epsilon * stddev2

# define placeholder for inputs to network
xs1 = tf.placeholder(tf.float32, [1, 1])
ys1 = tf.placeholder(tf.float32, [1, 2])



#loss=tf.reduce_mean(tf.square(predictValue-ys))
k0=0
loss20=tf.constant(0.0)
#for i in range(Dz-1):
#    predictValue0=RBF(mean[i,:],centerSize,outSize,K,0)
#    loss20=loss20+gammasPlace[i,k0]*tf.reduce_mean(tf.square(predictValue0-ys))


predictValue20=RBF(input_sample,centerSize,outSize,K,0)
temp20=gammasPlace[:,k0]*tf.reduce_sum(tf.square(predictValue20-ys),1,keep_dims=True)
loss20=tf.reduce_mean(temp20)*scale

tempU0=meanPlace[k0,:]#tf.to_float(tf.constant(uk[k,:]))
loss10=(tempU0-mean)
tempB0=BPlace[:,:,k0]#tf.to_float(tf.constant(wk[:,:,k]))
tempPro0=tf.matmul(loss10,tempB0)
tempA0=aPlace[k0]#tf.to_float(tf.constant(params['a'][k]))
loss0=gammasPlace[:,k0]*(tempA0*tempPro0*loss10+tempA0*tempB0*stddev-0.5*tf.log(stddev))/(1e5)
lossSum0=tf.reduce_mean(loss0)
lossMean0=tf.reduce_mean(tempA0*tempB0*stddev)
train_step0 = tf.train.GradientDescentOptimizer(learnRate1).minimize(lossSum0)
train_step01 = tf.train.GradientDescentOptimizer(learnRate2).minimize(loss20)

k1=tf.constant(1)
loss21=tf.constant(0.0)
predictValue21=RBF(input_sample,centerSize,outSize,K,1)
temp21=gammasPlace[:,k1]*tf.reduce_sum(tf.square(predictValue21-ys),1,keep_dims=True)
loss21=tf.reduce_mean(temp21)*scale

tempU1=meanPlace[k1,:]#tf.to_float(tf.constant(uk[k,:]))
loss11=(tempU1-mean)
tempB1=BPlace[:,:,k1]#tf.to_float(tf.constant(wk[:,:,k]))
tempPro1=tf.matmul(loss11,tempB1)
tempA1=aPlace[k1]#tf.to_float(tf.constant(params['a'][k]))
loss1=gammasPlace[:,k1]*(tempA1*tempPro1*loss11+tempA1*tempB1*stddev-0.5*tf.log(stddev))/(1e5)
lossSum1=tf.reduce_mean(loss1)
train_step1 = tf.train.GradientDescentOptimizer(learnRate1).minimize(lossSum1)
train_step1111 = tf.train.GradientDescentOptimizer(learnRate2).minimize(loss21+loss20)

# k2=tf.constant(2)
# loss22=tf.constant(0.0)
# predictValue22=RBF(input_sample,centerSize,outSize,K,2)
# temp22=gammasPlace[:,k2]*tf.reduce_sum(tf.square(predictValue22-ys),1,keep_dims=True)
# loss22=tf.reduce_mean(temp22)*scale
#
# tempU2=meanPlace[k2,:]#tf.to_float(tf.constant(uk[k,:]))
# loss12=(tempU2-mean)
# tempB2=BPlace[:,:,k2]#tf.to_float(tf.constant(wk[:,:,k]))
# tempPro2=tf.matmul(loss12,tempB2)
# tempA2=aPlace[k2]#tf.to_float(tf.constant(params['a'][k]))
# loss2=(tempA2*tf.reduce_sum(tempPro2*loss12,1,keep_dims=True)+tempA2*tempB2*stddev-0.5*tf.log(stddev))/(1e5)
# lossSum2=tf.reduce_mean(loss2)
# train_step2 = tf.train.GradientDescentOptimizer(learnRate1).minimize(lossSum2)
# train_step21 = tf.train.GradientDescentOptimizer(learnRate2).minimize(loss22)

# k3=tf.constant(3)
# loss23=tf.constant(0.0)
# predictValue23=RBF(input_sample,centerSize,outSize,K,3)
# temp23=gammasPlace[:,k3]*tf.reduce_sum(tf.square(predictValue23-ys),1,keep_dims=True)
# loss23=tf.reduce_mean(temp23)*scale
#
# tempU3=meanPlace[k3,:]#tf.to_float(tf.constant(uk[k,:]))
# loss13=(tempU3-mean)
# tempB3=BPlace[:,:,k3]#tf.to_float(tf.constant(wk[:,:,k]))
# tempPro3=tf.matmul(loss13,tempB3)
# tempA3=aPlace[k3]#tf.to_float(tf.constant(params['a'][k]))
# loss3=(tempA3*tf.reduce_sum(tempPro3*loss13,1,keep_dims=True)+tempA3*tempB3*stddev-0.5*tf.log(stddev))/(1e5)
# lossSum3=tf.reduce_mean(loss3)
# train_step3 = tf.train.GradientDescentOptimizer(learnRate1).minimize(lossSum3)
# train_step31 = tf.train.GradientDescentOptimizer(learnRate2).minimize(loss23)
#
#
# k4=tf.constant(4)
# loss24=tf.constant(0.0)
# predictValue24=RBF(input_sample,centerSize,outSize,K,4)
# temp24=gammasPlace[:,k4]*tf.reduce_sum(tf.square(predictValue24-ys),1,keep_dims=True)
# loss24=tf.reduce_mean(temp24)*scale
#
# tempU4=meanPlace[k4,:]#tf.to_float(tf.constant(uk[k,:]))
# loss14=(tempU4-mean)
# tempB4=BPlace[:,:,k4]#tf.to_float(tf.constant(wk[:,:,k]))
# tempPro4=tf.matmul(loss14,tempB4)
# tempA4=aPlace[k4]#tf.to_float(tf.constant(params['a'][k]))
# loss4=(tempA4*tf.reduce_sum(tempPro4*loss14,1,keep_dims=True)+tempA4*tempB4*stddev-0.5*tf.log(stddev))/(1e5)
# lossSum4=tf.reduce_mean(loss4)
# train_step4 = tf.train.GradientDescentOptimizer(learnRate1).minimize(lossSum4)
# train_step41 = tf.train.GradientDescentOptimizer(learnRate2).minimize(loss24)

# k5=tf.constant(5)
# loss25=tf.constant(0.0)
# predictValue25=RBF(input_sample,centerSize,outSize,K,5)
# temp25=gammasPlace[:,k5]*tf.reduce_sum(tf.square(predictValue25-ys),1,keep_dims=True)
# loss25=tf.reduce_mean(temp25)*scale
#
# tempU5=meanPlace[k5,:]#tf.to_float(tf.constant(uk[k,:]))
# loss15=(tempU5-mean)
# tempB5=BPlace[:,:,k5]#tf.to_float(tf.constant(wk[:,:,k]))
# tempPro5=tf.matmul(loss15,tempB5)
# tempA5=aPlace[k5]#tf.to_float(tf.constant(params['a'][k]))
# loss5=(tempA5*tf.reduce_sum(tempPro5*loss15,1,keep_dims=True)+tempA5*tempB5*stddev-0.5*tf.log(stddev))/(1e5)
# lossSum5=tf.reduce_mean(loss5)
# train_step5 = tf.train.GradientDescentOptimizer(learnRate1).minimize(lossSum5)
# train_step51 = tf.train.GradientDescentOptimizer(learnRate2).minimize(loss25)
#
# k6=tf.constant(6)
# loss26=tf.constant(0.0)
# predictValue26=RBF(input_sample,centerSize,outSize,K,6)
# temp26=gammasPlace[:,k6]*tf.reduce_sum(tf.square(predictValue26-ys),1,keep_dims=True)
# loss26=tf.reduce_mean(temp26)*scale
#
# tempU6=meanPlace[k6,:]#tf.to_float(tf.constant(uk[k,:]))
# loss16=(tempU6-mean)
# tempB6=BPlace[:,:,k6]#tf.to_float(tf.constant(wk[:,:,k]))
# tempPro6=tf.matmul(loss16,tempB6)
# tempA6=aPlace[k6]#tf.to_float(tf.constant(params['a'][k]))
# loss6=(tempA6*tf.reduce_sum(tempPro6*loss16,1,keep_dims=True)+tempA6*tempB6*stddev-0.5*tf.log(stddev))/(1e5)
# lossSum6=tf.reduce_mean(loss6)
# train_step6 = tf.train.GradientDescentOptimizer(learnRate1).minimize(lossSum6)
# train_step61 = tf.train.GradientDescentOptimizer(learnRate2).minimize(loss26)
#
#
# k7=tf.constant(7)
# loss27=tf.constant(0.0)
# predictValue27=RBF(input_sample,centerSize,outSize,K,7)
# temp27=gammasPlace[:,k7]*tf.reduce_sum(tf.square(predictValue27-ys),1,keep_dims=True)
# loss27=tf.reduce_mean(temp27)*scale
#
# tempU7=meanPlace[k7,:]#tf.to_float(tf.constant(uk[k,:]))
# loss17=(tempU7-mean)
# tempB7=BPlace[:,:,k7]#tf.to_float(tf.constant(wk[:,:,k]))
# tempPro7=tf.matmul(loss17,tempB7)
# tempA7=aPlace[k7]#tf.to_float(tf.constant(params['a'][k]))
# loss7=(tempA7*tf.reduce_sum(tempPro7*loss17,1,keep_dims=True)+tempA7*tempB7*stddev-0.5*tf.log(stddev))/(1e5)
# lossSum7=tf.reduce_mean(loss7)
# train_step7 = tf.train.GradientDescentOptimizer(learnRate1).minimize(lossSum7)
# train_step71 = tf.train.GradientDescentOptimizer(learnRate2).minimize(loss27)
#
# k8=tf.constant(8)
# loss28=tf.constant(0.0)
# predictValue28=RBF(input_sample,centerSize,outSize,K,8)
# temp28=gammasPlace[:,k8]*tf.reduce_sum(tf.square(predictValue28-ys),1,keep_dims=True)
# loss28=tf.reduce_mean(temp28)*scale
#
# tempU8=meanPlace[k8,:]#tf.to_float(tf.constant(uk[k,:]))
# loss18=(tempU8-mean)
# tempB8=BPlace[:,:,k8]#tf.to_float(tf.constant(wk[:,:,k]))
# tempPro8=tf.matmul(loss18,tempB8)
# tempA8=aPlace[k8]#tf.to_float(tf.constant(params['a'][k]))
# loss8=(tempA8*tf.reduce_sum(tempPro8*loss18,1,keep_dims=True)+tempA8*tempB8*stddev-0.5*tf.log(stddev))/(1e5)
# lossSum8=tf.reduce_mean(loss8)
# train_step8 = tf.train.GradientDescentOptimizer(learnRate1).minimize(lossSum8)
# train_step81 = tf.train.GradientDescentOptimizer(learnRate2).minimize(loss28)
#
# k9=tf.constant(9)
# loss29=tf.constant(0.0)
# predictValue29=RBF(input_sample,centerSize,outSize,K,9)
# temp29=gammasPlace[:,k9]*tf.reduce_sum(tf.square(predictValue29-ys),1,keep_dims=True)
# loss29=tf.reduce_mean(temp29)*scale
#
# tempU9=meanPlace[k9,:]#tf.to_float(tf.constant(uk[k,:]))
# loss19=(tempU9-mean)
# tempB9=BPlace[:,:,k9]#tf.to_float(tf.constant(wk[:,:,k]))
# tempPro9=tf.matmul(loss19,tempB9)
# tempA9=aPlace[k9]#tf.to_float(tf.constant(params['a'][k]))
# loss9=(tempA9*tf.reduce_sum(tempPro9*loss19,1,keep_dims=True)+tempA9*tempB9*stddev-0.5*tf.log(stddev))/(1e5)
# lossSum9=tf.reduce_mean(loss9)
# train_step9 = tf.train.GradientDescentOptimizer(learnRate1).minimize(lossSum9)
# train_step91 = tf.train.GradientDescentOptimizer(learnRate2).minimize(loss29)

# k10=tf.constant(10)
# loss210=tf.constant(0.0)
# predictValue210=RBF(input_sample,centerSize,outSize,K,10)
# temp210=gammasPlace[:,k10]*tf.reduce_sum(tf.square(predictValue210-ys),1,keep_dims=True)
# loss210=tf.reduce_mean(temp210)*scale
#
# tempU10=meanPlace[k10,:]#tf.to_float(tf.constant(uk[k,:]))
# loss110=(tempU10-mean)
# tempB10=BPlace[:,:,k10]#tf.to_float(tf.constant(wk[:,:,k]))
# tempPro10=tf.matmul(loss110,tempB10)
# tempA10=aPlace[k10]#tf.to_float(tf.constant(params['a'][k]))
# loss10=(tempA10*tf.reduce_sum(tempPro10*loss110,1,keep_dims=True)+tempA10*tempB10*stddev-0.5*tf.log(stddev))/(1e5)
# lossSum10=tf.reduce_mean(loss10)
# train_step10 = tf.train.GradientDescentOptimizer(learnRate1).minimize(lossSum10)
# train_step101 = tf.train.GradientDescentOptimizer(learnRate2).minimize(loss210)
#
#
# k11=tf.constant(11)
# loss211=tf.constant(0.0)
# predictValue211=RBF(input_sample,centerSize,outSize,K,11)
# temp211=gammasPlace[:,k11]*tf.reduce_sum(tf.square(predictValue211-ys),1,keep_dims=True)
# loss211=tf.reduce_mean(temp211)*scale
#
# tempU11=meanPlace[k11,:]#tf.to_float(tf.constant(uk[k,:]))
# loss111=(tempU11-mean)
# tempB11=BPlace[:,:,k11]#tf.to_float(tf.constant(wk[:,:,k]))
# tempPro11=tf.matmul(loss111,tempB11)
# tempA11=aPlace[k11]#tf.to_float(tf.constant(params['a'][k]))
# loss11=(tempA11*tf.reduce_sum(tempPro11*loss111,1,keep_dims=True)+tempA11*tempB11*stddev-0.5*tf.log(stddev))/(1e5)
# lossSum11=tf.reduce_mean(loss11)
# train_step11 = tf.train.GradientDescentOptimizer(learnRate1).minimize(lossSum11)
# train_step111 = tf.train.GradientDescentOptimizer(learnRate2).minimize(loss211)
#
#
# k12=tf.constant(12)
# loss212=tf.constant(0.0)
# predictValue212=RBF(input_sample,centerSize,outSize,K,12)
# temp212=gammasPlace[:,k12]*tf.reduce_sum(tf.square(predictValue212-ys),1,keep_dims=True)
# loss212=tf.reduce_mean(temp212)*scale
#
# tempU12=meanPlace[k12,:]#tf.to_float(tf.constant(uk[k,:]))
# loss112=(tempU12-mean)
# tempB12=BPlace[:,:,k12]#tf.to_float(tf.constant(wk[:,:,k]))
# tempPro12=tf.matmul(loss112,tempB12)
# tempA12=aPlace[k12]#tf.to_float(tf.constant(params['a'][k]))
# loss12=(tempA12*tf.reduce_sum(tempPro12*loss112,1,keep_dims=True)+tempA12*tempB12*stddev-0.5*tf.log(stddev))/(1e5)
# lossSum12=tf.reduce_mean(loss12)
# train_step12 = tf.train.GradientDescentOptimizer(learnRate1).minimize(lossSum12)
# train_step121 = tf.train.GradientDescentOptimizer(learnRate2).minimize(loss212)
#
# k13=tf.constant(13)
# loss213=tf.constant(0.0)
# predictValue213=RBF(input_sample,centerSize,outSize,K,13)
# temp213=gammasPlace[:,k13]*tf.reduce_sum(tf.square(predictValue213-ys),1,keep_dims=True)
# loss213=tf.reduce_mean(temp213)*scale
#
# tempU13=meanPlace[k13,:]#tf.to_float(tf.constant(uk[k,:]))
# loss113=(tempU13-mean)
# tempB13=BPlace[:,:,k13]#tf.to_float(tf.constant(wk[:,:,k]))
# tempPro13=tf.matmul(loss113,tempB13)
# tempA13=aPlace[k13]#tf.to_float(tf.constant(params['a'][k]))
# loss13=(tempA13*tf.reduce_sum(tempPro13*loss113,1,keep_dims=True)+tempA13*tempB13*stddev-0.5*tf.log(stddev))/(1e5)
# lossSum13=tf.reduce_mean(loss13)
# train_step13 = tf.train.GradientDescentOptimizer(learnRate1).minimize(lossSum13)
# train_step131 = tf.train.GradientDescentOptimizer(learnRate2).minimize(loss213)
#
#
#
# k14=tf.constant(14)
# loss214=tf.constant(0.0)
# predictValue214=RBF(input_sample,centerSize,outSize,K,14)
# temp214=gammasPlace[:,k14]*tf.reduce_sum(tf.square(predictValue214-ys),1,keep_dims=True)
# loss214=tf.reduce_mean(temp214)*scale
#
# tempU14=meanPlace[k14,:]#tf.to_float(tf.constant(uk[k,:]))
# loss114=(tempU14-mean)
# tempB14=BPlace[:,:,k14]#tf.to_float(tf.constant(wk[:,:,k]))
# tempPro14=tf.matmul(loss114,tempB14)
# tempA14=aPlace[k14]#tf.to_float(tf.constant(params['a'][k]))
# loss14=(tempA14*tf.reduce_sum(tempPro14*loss114,1,keep_dims=True)+tempA14*tempB14*stddev-0.5*tf.log(stddev))/(1e5)
# lossSum14=tf.reduce_mean(loss14)
# train_step14 = tf.train.GradientDescentOptimizer(learnRate1).minimize(lossSum14)
# train_step141 = tf.train.GradientDescentOptimizer(learnRate2).minimize(loss214)
#
#
# k15=tf.constant(15)
# loss215=tf.constant(0.0)
# predictValue215=RBF(input_sample,centerSize,outSize,K,15)
# temp215=gammasPlace[:,k15]*tf.reduce_sum(tf.square(predictValue215-ys),1,keep_dims=True)
# loss215=tf.reduce_mean(temp215)*scale
#
# tempU15=meanPlace[k15,:]#tf.to_float(tf.constant(uk[k,:]))
# loss115=(tempU15-mean)
# tempB15=BPlace[:,:,k15]#tf.to_float(tf.constant(wk[:,:,k]))
# tempPro15=tf.matmul(loss115,tempB15)
# tempA15=aPlace[k15]#tf.to_float(tf.constant(params['a'][k]))
# loss15=(tempA15*tf.reduce_sum(tempPro15*loss115,1,keep_dims=True)+tempA15*tempB15*stddev-0.5*tf.log(stddev))/(1e5)
# lossSum15=tf.reduce_mean(loss15)
# train_step15 = tf.train.GradientDescentOptimizer(learnRate1).minimize(lossSum15)
# train_step151 = tf.train.GradientDescentOptimizer(learnRate2).minimize(loss215)
#
#
# k16=tf.constant(16)
# loss216=tf.constant(0.0)
# predictValue216=RBF(input_sample,centerSize,outSize,K,16)
# temp216=gammasPlace[:,k16]*tf.reduce_sum(tf.square(predictValue216-ys),1,keep_dims=True)
# loss216=tf.reduce_mean(temp216)*scale
#
# tempU16=meanPlace[k16,:]#tf.to_float(tf.constant(uk[k,:]))
# loss116=(tempU16-mean)
# tempB16=BPlace[:,:,k16]#tf.to_float(tf.constant(wk[:,:,k]))
# tempPro16=tf.matmul(loss116,tempB16)
# tempA16=aPlace[k16]#tf.to_float(tf.constant(params['a'][k]))
# loss16=(tempA16*tf.reduce_sum(tempPro16*loss116,1,keep_dims=True)+tempA16*tempB16*stddev-0.5*tf.log(stddev))/(1e5)
# lossSum16=tf.reduce_mean(loss16)
# train_step16 = tf.train.GradientDescentOptimizer(learnRate1).minimize(lossSum16)
# train_step161 = tf.train.GradientDescentOptimizer(learnRate2).minimize(loss216)
#
#
# k17=tf.constant(17)
# loss217=tf.constant(0.0)
# predictValue217=RBF(input_sample,centerSize,outSize,K,17)
# temp217=gammasPlace[:,k17]*tf.reduce_sum(tf.square(predictValue217-ys),1,keep_dims=True)
# loss217=tf.reduce_mean(temp217)*scale
#
# tempU17=meanPlace[k17,:]#tf.to_float(tf.constant(uk[k,:]))
# loss117=(tempU17-mean)
# tempB17=BPlace[:,:,k17]#tf.to_float(tf.constant(wk[:,:,k]))
# tempPro17=tf.matmul(loss117,tempB17)
# tempA17=aPlace[k17]#tf.to_float(tf.constant(params['a'][k]))
# loss17=(tempA17*tf.reduce_sum(tempPro17*loss117,1,keep_dims=True)+tempA17*tempB17*stddev-0.5*tf.log(stddev))/(1e5)
# lossSum17=tf.reduce_mean(loss17)
# train_step17 = tf.train.GradientDescentOptimizer(learnRate1).minimize(lossSum17)
# train_step171 = tf.train.GradientDescentOptimizer(learnRate2).minimize(loss217)
#
#
# k18=tf.constant(18)
# loss218=tf.constant(0.0)
# predictValue218=RBF(input_sample,centerSize,outSize,K,18)
# temp218=gammasPlace[:,k18]*tf.reduce_sum(tf.square(predictValue218-ys),1,keep_dims=True)
# loss218=tf.reduce_mean(temp218)*scale
#
# tempU18=meanPlace[k18,:]#tf.to_float(tf.constant(uk[k,:]))
# loss118=(tempU18-mean)
# tempB18=BPlace[:,:,k18]#tf.to_float(tf.constant(wk[:,:,k]))
# tempPro18=tf.matmul(loss118,tempB18)
# tempA18=aPlace[k18]#tf.to_float(tf.constant(params['a'][k]))
# loss18=(tempA18*tf.reduce_sum(tempPro18*loss118,1,keep_dims=True)+tempA18*tempB18*stddev-0.5*tf.log(stddev))/(1e5)
# lossSum18=tf.reduce_mean(loss18)
# train_step18 = tf.train.GradientDescentOptimizer(learnRate1).minimize(lossSum18)
# train_step181 = tf.train.GradientDescentOptimizer(learnRate2).minimize(loss218)
#
#
# k19=tf.constant(19)
# loss219=tf.constant(0.0)
# predictValue219=RBF(input_sample,centerSize,outSize,K,19)
# temp219=gammasPlace[:,k19]*tf.reduce_sum(tf.square(predictValue219-ys),1,keep_dims=True)
# loss219=tf.reduce_mean(temp219)*scale
#
# tempU19=meanPlace[k19,:]#tf.to_float(tf.constant(uk[k,:]))
# loss119=(tempU19-mean)
# tempB19=BPlace[:,:,k19]#tf.to_float(tf.constant(wk[:,:,k]))
# tempPro19=tf.matmul(loss119,tempB19)
# tempA19=aPlace[k19]#tf.to_float(tf.constant(params['a'][k]))
# loss19=(tempA19*tf.reduce_sum(tempPro19*loss119,1,keep_dims=True)+tempA19*tempB19*stddev-0.5*tf.log(stddev))/(1e5)
# lossSum19=tf.reduce_mean(loss19)
# train_step19 = tf.train.GradientDescentOptimizer(learnRate1).minimize(lossSum19)
# train_step191 = tf.train.GradientDescentOptimizer(learnRate2).minimize(loss219)

# k30=tf.constant(20)
# loss230=tf.constant(0.0)
# predictValue230=RBF(input_sample,centerSize,outSize,K,20)
# temp230=gammasPlace[:,k30]*tf.reduce_sum(tf.square(predictValue230-ys),1,keep_dims=True)
# loss230=tf.reduce_mean(temp230)*scale
#
# tempU30=meanPlace[k30,:]#tf.to_float(tf.constant(uk[k,:]))
# loss130=(tempU30-mean)
# tempB30=BPlace[:,:,k30]#tf.to_float(tf.constant(wk[:,:,k]))
# tempPro30=tf.matmul(loss130,tempB30)
# tempA30=aPlace[k30]#tf.to_float(tf.constant(params['a'][k]))
# loss30=(tempA30*tf.reduce_sum(tempPro30*loss130,1,keep_dims=True)+tempA30*tempB30*stddev-0.5*tf.log(stddev))/(1e5)
# lossSum30=tf.reduce_mean(loss30)
# train_step30 = tf.train.GradientDescentOptimizer(learnRate1).minimize(lossSum30)
# train_step301 = tf.train.GradientDescentOptimizer(learnRate2).minimize(loss230)
#
# k31=tf.constant(21)
# loss231=tf.constant(0.0)
# predictValue231=RBF(input_sample,centerSize,outSize,K,21)
# temp231=gammasPlace[:,k31]*tf.reduce_sum(tf.square(predictValue231-ys),1,keep_dims=True)
# loss231=tf.reduce_mean(temp231)*scale
#
# tempU31=meanPlace[k31,:]#tf.to_float(tf.constant(uk[k,:]))
# loss131=(tempU31-mean)
# tempB31=BPlace[:,:,k31]#tf.to_float(tf.constant(wk[:,:,k]))
# tempPro31=tf.matmul(loss131,tempB31)
# tempA31=aPlace[k31]#tf.to_float(tf.constant(params['a'][k]))
# loss31=(tempA31*tf.reduce_sum(tempPro31*loss131,1,keep_dims=True)+tempA31*tempB31*stddev-0.5*tf.log(stddev))/(1e5)
# lossSum31=tf.reduce_mean(loss31)
# train_step31 = tf.train.GradientDescentOptimizer(learnRate1).minimize(lossSum31)
# train_step311 = tf.train.GradientDescentOptimizer(learnRate2).minimize(loss231)
#
# k32=tf.constant(22)
# loss232=tf.constant(0.0)
# predictValue232=RBF(input_sample,centerSize,outSize,K,22)
# temp232=gammasPlace[:,k32]*tf.reduce_sum(tf.square(predictValue232-ys),1,keep_dims=True)
# loss232=tf.reduce_mean(temp232)*scale
#
# tempU32=meanPlace[k32,:]#tf.to_float(tf.constant(uk[k,:]))
# loss132=(tempU32-mean)
# tempB32=BPlace[:,:,k32]#tf.to_float(tf.constant(wk[:,:,k]))
# tempPro32=tf.matmul(loss132,tempB32)
# tempA32=aPlace[k32]#tf.to_float(tf.constant(params['a'][k]))
# loss32=(tempA32*tf.reduce_sum(tempPro32*loss132,1,keep_dims=True)+tempA32*tempB32*stddev-0.5*tf.log(stddev))/(1e5)
# lossSum32=tf.reduce_mean(loss32)
# train_step32 = tf.train.GradientDescentOptimizer(learnRate1).minimize(lossSum32)
# train_step321 = tf.train.GradientDescentOptimizer(learnRate2).minimize(loss232)
#
# k33=tf.constant(23)
# loss233=tf.constant(0.0)
# predictValue233=RBF(input_sample,centerSize,outSize,K,23)
# temp233=gammasPlace[:,k33]*tf.reduce_sum(tf.square(predictValue233-ys),1,keep_dims=True)
# loss233=tf.reduce_mean(temp233)*scale
#
# tempU33=meanPlace[k33,:]#tf.to_float(tf.constant(uk[k,:]))
# loss133=(tempU33-mean)
# tempB33=BPlace[:,:,k33]#tf.to_float(tf.constant(wk[:,:,k]))
# tempPro33=tf.matmul(loss133,tempB33)
# tempA33=aPlace[k33]#tf.to_float(tf.constant(params['a'][k]))
# loss33=(tempA33*tf.reduce_sum(tempPro33*loss133,1,keep_dims=True)+tempA33*tempB33*stddev-0.5*tf.log(stddev))/(1e5)
# lossSum33=tf.reduce_mean(loss33)
# train_step33 = tf.train.GradientDescentOptimizer(learnRate1).minimize(lossSum33)
# train_step331 = tf.train.GradientDescentOptimizer(learnRate2).minimize(loss233)
#
# k34=tf.constant(24)
# loss234=tf.constant(0.0)
# predictValue234=RBF(input_sample,centerSize,outSize,K,24)
# temp234=gammasPlace[:,k34]*tf.reduce_sum(tf.square(predictValue234-ys),1,keep_dims=True)
# loss234=tf.reduce_mean(temp234)*scale
#
# tempU34=meanPlace[k34,:]#tf.to_float(tf.constant(uk[k,:]))
# loss134=(tempU34-mean)
# tempB34=BPlace[:,:,k34]#tf.to_float(tf.constant(wk[:,:,k]))
# tempPro34=tf.matmul(loss134,tempB34)
# tempA34=aPlace[k34]#tf.to_float(tf.constant(params['a'][k]))
# loss34=(tempA34*tf.reduce_sum(tempPro34*loss134,1,keep_dims=True)+tempA34*tempB34*stddev-0.5*tf.log(stddev))/(1e5)
# lossSum34=tf.reduce_mean(loss34)
# train_step34 = tf.train.GradientDescentOptimizer(learnRate1).minimize(lossSum34)
# train_step341 = tf.train.GradientDescentOptimizer(learnRate2).minimize(loss234)
#
# k35=tf.constant(25)
# loss235=tf.constant(0.0)
# predictValue235=RBF(input_sample,centerSize,outSize,K,25)
# temp235=gammasPlace[:,k35]*tf.reduce_sum(tf.square(predictValue235-ys),1,keep_dims=True)
# loss235=tf.reduce_mean(temp235)*scale
#
# tempU35=meanPlace[k35,:]#tf.to_float(tf.constant(uk[k,:]))
# loss135=(tempU35-mean)
# tempB35=BPlace[:,:,k35]#tf.to_float(tf.constant(wk[:,:,k]))
# tempPro35=tf.matmul(loss135,tempB35)
# tempA35=aPlace[k35]#tf.to_float(tf.constant(params['a'][k]))
# loss35=(tempA35*tf.reduce_sum(tempPro35*loss135,1,keep_dims=True)+tempA35*tempB35*stddev-0.5*tf.log(stddev))/(1e5)
# lossSum35=tf.reduce_mean(loss35)
# train_step35 = tf.train.GradientDescentOptimizer(learnRate1).minimize(lossSum35)
# train_step351 = tf.train.GradientDescentOptimizer(learnRate2).minimize(loss235)
#
# k36=tf.constant(26)
# loss236=tf.constant(0.0)
# predictValue236=RBF(input_sample,centerSize,outSize,K,26)
# temp236=gammasPlace[:,k36]*tf.reduce_sum(tf.square(predictValue236-ys),1,keep_dims=True)
# loss236=tf.reduce_mean(temp236)*scale
#
# tempU36=meanPlace[k36,:]#tf.to_float(tf.constant(uk[k,:]))
# loss136=(tempU36-mean)
# tempB36=BPlace[:,:,k36]#tf.to_float(tf.constant(wk[:,:,k]))
# tempPro36=tf.matmul(loss136,tempB36)
# tempA36=aPlace[k36]#tf.to_float(tf.constant(params['a'][k]))
# loss36=(tempA36*tf.reduce_sum(tempPro36*loss136,1,keep_dims=True)+tempA36*tempB36*stddev-0.5*tf.log(stddev))/(1e5)
# lossSum36=tf.reduce_mean(loss36)
# train_step36 = tf.train.GradientDescentOptimizer(learnRate1).minimize(lossSum36)
# train_step361 = tf.train.GradientDescentOptimizer(learnRate2).minimize(loss236)
#
# k37=tf.constant(27)
# loss237=tf.constant(0.0)
# predictValue237=RBF(input_sample,centerSize,outSize,K,27)
# temp237=gammasPlace[:,k37]*tf.reduce_sum(tf.square(predictValue237-ys),1,keep_dims=True)
# loss237=tf.reduce_mean(temp237)*scale
#
# tempU37=meanPlace[k37,:]#tf.to_float(tf.constant(uk[k,:]))
# loss137=(tempU37-mean)
# tempB37=BPlace[:,:,k37]#tf.to_float(tf.constant(wk[:,:,k]))
# tempPro37=tf.matmul(loss137,tempB37)
# tempA37=aPlace[k37]#tf.to_float(tf.constant(params['a'][k]))
# loss37=(tempA37*tf.reduce_sum(tempPro37*loss137,1,keep_dims=True)+tempA37*tempB37*stddev-0.5*tf.log(stddev))/(1e5)
# lossSum37=tf.reduce_mean(loss37)
# train_step37 = tf.train.GradientDescentOptimizer(learnRate1).minimize(lossSum37)
# train_step371 = tf.train.GradientDescentOptimizer(learnRate2).minimize(loss237)
#
# k38=tf.constant(28)
# loss238=tf.constant(0.0)
# predictValue238=RBF(input_sample,centerSize,outSize,K,28)
# temp238=gammasPlace[:,k38]*tf.reduce_sum(tf.square(predictValue238-ys),1,keep_dims=True)
# loss238=tf.reduce_mean(temp238)*scale
#
# tempU38=meanPlace[k38,:]#tf.to_float(tf.constant(uk[k,:]))
# loss138=(tempU38-mean)
# tempB38=BPlace[:,:,k38]#tf.to_float(tf.constant(wk[:,:,k]))
# tempPro38=tf.matmul(loss138,tempB38)
# tempA38=aPlace[k38]#tf.to_float(tf.constant(params['a'][k]))
# loss38=(tempA38*tf.reduce_sum(tempPro38*loss138,1,keep_dims=True)+tempA38*tempB38*stddev-0.5*tf.log(stddev))/(1e5)
# lossSum38=tf.reduce_mean(loss38)
# train_step38 = tf.train.GradientDescentOptimizer(learnRate1).minimize(lossSum38)
# train_step381 = tf.train.GradientDescentOptimizer(learnRate2).minimize(loss238)
#
#
# k39=tf.constant(29)
# loss239=tf.constant(0.0)
# predictValue239=RBF(input_sample,centerSize,outSize,K,29)
# temp239=gammasPlace[:,k39]*tf.reduce_sum(tf.square(predictValue239-ys),1,keep_dims=True)
# loss239=tf.reduce_mean(temp239)*scale
#
# tempU39=meanPlace[k39,:]#tf.to_float(tf.constant(uk[k,:]))
# loss139=(tempU39-mean)
# tempB39=BPlace[:,:,k39]#tf.to_float(tf.constant(wk[:,:,k]))
# tempPro39=tf.matmul(loss139,tempB39)
# tempA39=aPlace[k39]#tf.to_float(tf.constant(params['a'][k]))
# loss39=(tempA39*tf.reduce_sum(tempPro39*loss139,1,keep_dims=True)+tempA39*tempB39*stddev-0.5*tf.log(stddev))/(1e5)
# lossSum39=tf.reduce_mean(loss39)
# train_step39 = tf.train.GradientDescentOptimizer(learnRate1).minimize(lossSum39)
# train_step391 = tf.train.GradientDescentOptimizer(learnRate2).minimize(loss239)


xsRef = tf.placeholder(tf.float32, [None, hiden_size])
ys1 = tf.placeholder(tf.float32, [None, outSize])
predictValueRef=RBF(xsRef,50,outSize,K,0)
tempRef=tf.reduce_mean(tf.square(predictValueRef-ys),1,keep_dims=True)
lossRef=tf.reduce_mean(tempRef)
train_stepRef = tf.train.GradientDescentOptimizer(0.1).minimize(lossRef)

xsRef1 = tf.placeholder(tf.float32, [None, hiden_size])
predictValueRef1=RBF(xsRef1,50,outSize,K,0)
tempRef1=tf.reduce_mean(tf.square(predictValueRef1-ys1),1,keep_dims=True)
lossRef1=tf.reduce_mean(tempRef1)
train_stepRef1 = tf.train.GradientDescentOptimizer(0.1).minimize(lossRef1)

init = tf.global_variables_initializer()


run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth=True
run_config.gpu_options.per_process_gpu_memory_fraction = 1/10
sess = tf.Session(config=run_config)
sess.run(init)

#K=0.1;

#xTest1=np.arange(0,K,0.1)
#xTest1.shape=(xTest1.shape[0],1)
#xTest2=np.arange(0,K,0.1)
#xTest2.shape=(xTest2.shape[0],1)
#xTest=np.concatenate((xTest1,xTest2),axis=1)
#xTest.shape=(xTest.shape[0],1)
#xTest=data[0,:].reshape(1,2)

posMean=sess.run(mean,feed_dict={xs: data})
posCov=sess.run(stddev,feed_dict={xs: data})

params,gammas = vdpmm_init(posMean,K)
paramsGaussian,gammasGaussian = vdpmm_init(data,K)
uk=params['mean']


model_vars = tf.trainable_variables()
slim.model_analyzer.analyze_vars(model_vars, print_info=True)

while numits < maxits:
    posMean=sess.run(mean,feed_dict={xs: data})
    posCov=sess.run(stddev,feed_dict={xs: data})


    posGammas=sess.run(tf.reduce_sum(tf.square(predictValue20-ys),1,keep_dims=True), feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
    #posGammas=np.c_[posGammas,sess.run(tf.reduce_sum(tf.square(predictValue20-ys),1,keep_dims=True), feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})]
    posGammas=np.c_[posGammas,sess.run(tf.reduce_sum(tf.square(predictValue21-ys),1,keep_dims=True), feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})]
    # posGammas=np.c_[posGammas,sess.run(tf.reduce_sum(tf.square(predictValue22-ys),1,keep_dims=True), feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})]
    # posGammas=np.c_[posGammas,sess.run(tf.reduce_sum(tf.square(predictValue23-ys),1,keep_dims=True), feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})]
    # posGammas=np.c_[posGammas,sess.run(tf.reduce_sum(tf.square(predictValue24-ys),1,keep_dims=True), feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})]
    # posGammas=np.c_[posGammas,sess.run(tf.reduce_sum(tf.square(predictValue25-ys),1,keep_dims=True), feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})]
    # posGammas=np.c_[posGammas,sess.run(tf.reduce_sum(tf.square(predictValue26-ys),1,keep_dims=True), feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})]
    # posGammas=np.c_[posGammas,sess.run(tf.reduce_sum(tf.square(predictValue27-ys),1,keep_dims=True), feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})]
    # posGammas=np.c_[posGammas,sess.run(tf.reduce_sum(tf.square(predictValue28-ys),1,keep_dims=True), feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})]
    # posGammas=np.c_[posGammas,sess.run(tf.reduce_sum(tf.square(predictValue29-ys),1,keep_dims=True), feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})]
    # posGammas=np.c_[posGammas,sess.run(tf.reduce_sum(tf.square(predictValue210-ys),1,keep_dims=True), feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})]
    # posGammas=np.c_[posGammas,sess.run(tf.reduce_sum(tf.square(predictValue211-ys),1,keep_dims=True), feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})]
    # posGammas=np.c_[posGammas,sess.run(tf.reduce_sum(tf.square(predictValue212-ys),1,keep_dims=True), feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})]
    # posGammas=np.c_[posGammas,sess.run(tf.reduce_sum(tf.square(predictValue213-ys),1,keep_dims=True), feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})]
    # posGammas=np.c_[posGammas,sess.run(tf.reduce_sum(tf.square(predictValue214-ys),1,keep_dims=True), feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})]
    # posGammas=np.c_[posGammas,sess.run(tf.reduce_sum(tf.square(predictValue215-ys),1,keep_dims=True), feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})]
    # posGammas=np.c_[posGammas,sess.run(tf.reduce_sum(tf.square(predictValue216-ys),1,keep_dims=True), feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})]
    # posGammas=np.c_[posGammas,sess.run(tf.reduce_sum(tf.square(predictValue217-ys),1,keep_dims=True), feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})]
    # posGammas=np.c_[posGammas,sess.run(tf.reduce_sum(tf.square(predictValue218-ys),1,keep_dims=True), feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})]
    # posGammas=np.c_[posGammas,sess.run(tf.reduce_sum(tf.square(predictValue219-ys),1,keep_dims=True), feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})]
    # posGammas=np.c_[posGammas,sess.run(tf.reduce_sum(tf.square(predictValue230-ys),1,keep_dims=True), feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})]
    # posGammas=np.c_[posGammas,sess.run(tf.reduce_sum(tf.square(predictValue231-ys),1,keep_dims=True), feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})]
    # posGammas=np.c_[posGammas,sess.run(tf.reduce_sum(tf.square(predictValue232-ys),1,keep_dims=True), feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})]
    # posGammas=np.c_[posGammas,sess.run(tf.reduce_sum(tf.square(predictValue233-ys),1,keep_dims=True), feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})]
    # posGammas=np.c_[posGammas,sess.run(tf.reduce_sum(tf.square(predictValue234-ys),1,keep_dims=True), feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})]
    # posGammas=np.c_[posGammas,sess.run(tf.reduce_sum(tf.square(predictValue235-ys),1,keep_dims=True), feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})]
    # posGammas=np.c_[posGammas,sess.run(tf.reduce_sum(tf.square(predictValue236-ys),1,keep_dims=True), feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})]
    # posGammas=np.c_[posGammas,sess.run(tf.reduce_sum(tf.square(predictValue237-ys),1,keep_dims=True), feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})]
    # posGammas=np.c_[posGammas,sess.run(tf.reduce_sum(tf.square(predictValue238-ys),1,keep_dims=True), feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})]
    # posGammas=np.c_[posGammas,sess.run(tf.reduce_sum(tf.square(predictValue239-ys),1,keep_dims=True), feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})]
    #
    #
    #
    paramsGaussian=vdpmm_maximizePlusGaussian(data,paramsGaussian,(1-lambdaPos)*gammas)
    params = vdpmm_maximizeCNN(posMean,params,lambdaPos*gammas,posCov);
    posGaussian=vdpmm_expectationPlusGaussian(data,paramsGaussian)
    gammas=vdpmm_expectationCNN(posMean,params,lambdaPos*posGammas+(1-lambdaPos)*posGaussian)

#    uk=params['mean']
#    wk=params['B']
#    lossPos=tf.constant(0.0)
#    for k in range(K):
#        tempU=tf.to_float(tf.constant(uk[k,:]))
#        loss1=(tempU-mean)
#        tempB=tf.to_float(tf.constant(wk[:,:,k]))
#        tempPro=tf.matmul(loss1,tempB)
#        tempA=tf.to_float(tf.constant(params['a'][k]))
#        loss=tempA*tf.reduce_sum(tempPro*loss1,1,keep_dims=True)+tempA*tempB*stddev-0.5*tf.log(stddev)
#        lossSum=tf.reduce_mean(loss)
#
#        lossPos=lossPos+lossSum
    #print(sess.run(lossPos, feed_dict={xs: data}))


    peace=10
    # print('loss',numits)
    for inte in range(50):
        lossS=0
        #lossS=sess.run(lossSum0, feed_dict={xs: data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        #print(lossS)

        sess.run(train_step0, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        sess.run(train_step1, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step2, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step3, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step4, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step5, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step6, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step7, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step8, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step9, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step10, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step11, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step12, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step13, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step14, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step15, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step16, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step17, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step18, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step19, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step30, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step31, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step32, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step33, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step34, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step35, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step36, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step37, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step38, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step39, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})


        #
        lossS=sess.run(lossSum0, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        lossS=lossS+sess.run(lossSum1, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(lossSum2, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(lossSum3, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(lossSum4, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(lossSum5, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(lossSum6, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(lossSum7, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(lossSum8, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(lossSum9, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(lossSum10, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(lossSum11, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(lossSum12, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(lossSum13, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(lossSum14, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(lossSum15, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(lossSum16, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(lossSum17, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(lossSum18, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(lossSum19, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # print(lossS)

        # lossS=sess.run(loss20, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(loss21, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(loss22, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(loss23, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(loss24, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(loss25, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(loss26, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})

        #lossS=lossS+sess.run(tf.reduce_mean(mean), feed_dict={xs: data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})

#        peaceK=int((Dz-1)/peace)
#        peaceMod=(Dz-1)%peace
#        for sli in range(peaceK):
#            tempData=data[sli*peace:(sli+1)*peace,:]
#            sess.run(train_step, feed_dict={xs: tempData})
#            lossS=lossS+sess.run(lossPos, feed_dict={xs: tempData})

        print(sess.run(lossMean0, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas}))
        # print(lossS)
    for inte in range(50):
        lossS=0
        #lossS=sess.run(lossSum0, feed_dict={xs: data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        #print(lossS)



        sess.run(train_step01, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        sess.run(train_step1111, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step21, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step31, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step41, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step51, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step61, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step71, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step81, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step91, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step101, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step111, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step121, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step131, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step141, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step151, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step161, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step171, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step181, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step191, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step301, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step311, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step321, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step331, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step341, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step351, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step361, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step371, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step381, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step391, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})


        lossS=sess.run(loss20, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        lossS=lossS+sess.run(loss21, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(loss22, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(loss23, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(loss24, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(loss25, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(loss26, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(loss27, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(loss28, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(loss29, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(loss210, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(loss211, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(loss212, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(loss213, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(loss214, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(loss215, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(loss216, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(loss217, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(loss218, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(loss219, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(loss230, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(loss231, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(loss232, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(loss233, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(loss234, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(loss235, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(loss236, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(loss237, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(loss238, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # lossS=lossS+sess.run(loss239, feed_dict={xs: data,ys:data,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})

        #
        # print(lossS)
    numits=numits+1

print(gammas);
temp=np.max(gammas,axis=1)
temp.shape=(Dz,1)
index1=np.where(temp==gammas)
color=np.unique(index1[1])


fig=plt.figure(1)
ax1=fig.add_subplot(111,projection='3d')

colorStore='rgbyck'
for i in range(Dz):
    cho=np.mod(index1[1][i],6)
    # plt.scatter(data[i,0],data[i,1],color=colorStore[cho])
    ax1.scatter(data[i,0],data[i,1],data[i,1],color=colorStore[cho])
sio.savemat('saveddata1.mat', {'data': data,'inputK': index1})

plt.show();