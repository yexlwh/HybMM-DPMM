# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 20:21:41 2017

@author: yexlwh
"""

from sklearn.cluster import KMeans
import numpy as np

def dataCombineX(data,K):
#    kmeans = KMeans(n_clusters=K, random_state=0).fit(data)
#    label=kmeans.labels_
#    
#    index=np.array([])
#    for k in range(K):
#        temp=np.where(label==(k))
#        index=np.append(index,temp[0][0])
    N,D=data.shape
    sortIndex=np.argsort(data[:,0],axis=0)
    
    sortSlice=int(N/K)
    x=np.zeros((N,1))
    
    for k in range(K):
        for j in range(sortSlice):
            x[sortIndex[k*sortSlice+j]]=k
    
    for i in range((K*sortSlice),N):
        x[sortIndex[i]]=K+1
    return x