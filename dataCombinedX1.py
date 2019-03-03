# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 12:09:12 2017

@author: yexlwh
"""

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt;

def dataCombineX(data,K):
    kmeans = KMeans(n_clusters=K, random_state=0).fit(data)
    center = kmeans.cluster_centers_
    label=kmeans.labels_
    sortCenterIndex=np.argsort(center[:,0],axis=0)
    
    N,D=center.shape
    flag=np.zeros((N,1))
    flag[sortCenterIndex[-1]]=1
    rem=sortCenterIndex[-1]
    
    for i in range(N):
        dist=np.sum((center[rem,:]-center)*(center[rem,:]-center),axis=1)
        dist[0]=np.Inf
        distIndex=np.argsort(dist,axis=0)
        for j in range(N):
            if flag[distIndex[j]]==0:
                flag[distIndex[j]]=flag[rem]+1
                rem=distIndex[j]
                break
            
    x=np.zeros(label.shape)
    for k in range(K):
        indexTemp=np.where(label==(k))
        x[indexTemp]=flag[k]

    # plt.figure(1)
    # colorStore='rgbyck'
    # for i in range(data.shape[0]-1):
    #     cho=int(np.mod(x[i],6))
    #     plt.scatter(data[i,0],data[i,1],color=colorStore[cho])
    # plt.show()
    return x,center