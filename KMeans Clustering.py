# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 02:29:57 2019

@author: Dilip
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#import dataset 
dataframe=pd.read_csv('Customers.csv')
x=dataframe.iloc[:,[3,4]].values

#using the elbow methord
from sklearn.cluster import KMeans
    
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++',random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title('The elbow methord')
plt.xlabel('Number of Cluster')
plt.ylabel('wcss')
plt.show()

#Fitting kmeans to the dataset
kmeans=KMeans(n_clusters=3, init='k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(x)

#visualize the cluster
plt.scatter(x[y_kmeans==0,0], x[y_kmeans==0,1], s=200, c='blue', label='Cluster1')
plt.scatter(x[y_kmeans==1,0], x[y_kmeans==1,1], s=200, c='green', label='Cluster2')
plt.scatter(x[y_kmeans==2,0], x[y_kmeans==2,1], s=200, c='yellow', label='Cluster3')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='cyan',label='Centroids')


plt.title('Clusters of customers')
plt.xlabel('Annual Income(K$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()






    
