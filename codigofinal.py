# -*- coding: utf-8 -*-
"""
Created on Sun nov 21 17:33:06 2021

@author: JP Zuluaga
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


df = pd.read_csv('database.csv', sep=',', header=0)
# Check si hay NaN en el data
NaNcont = df.isnull().sum().sum()

# limpio el dataframe
df.dropna(subset = ["CUST_ID"], inplace=True)
df.dropna(subset = ["BALANCE"], inplace=True)
df.dropna(subset = ["BALANCE_FREQUENCY"], inplace=True)
df.dropna(subset = ["PURCHASES"], inplace=True)
df.dropna(subset = ["ONEOFF_PURCHASES"], inplace=True)
df.dropna(subset = ["INSTALLMENTS_PURCHASES"], inplace=True)
df.dropna(subset = ["CASH_ADVANCE"], inplace=True)
df.dropna(subset = ["PURCHASES_FREQUENCY"], inplace=True)
df.dropna(subset = ["ONEOFF_PURCHASES_FREQUENCY"], inplace=True)
df.dropna(subset = ["PURCHASES_INSTALLMENTS_FREQUENCY"], inplace=True)
df.dropna(subset = ["CASH_ADVANCE_FREQUENCY"], inplace=True)
df.dropna(subset = ["CASH_ADVANCE_TRX"], inplace=True)
df.dropna(subset = ["PURCHASES_TRX"], inplace=True)
df.dropna(subset = ["CREDIT_LIMIT"], inplace=True)
df.dropna(subset = ["PAYMENTS"], inplace=True)
df.dropna(subset = ["MINIMUM_PAYMENTS"], inplace=True)
df.dropna(subset = ["PRC_FULL_PAYMENT"], inplace=True)
df.dropna(subset = ["TENURE"], inplace=True)

X = df.values

# Separo los cardholders 
cardHolders = X[:,0]
X = X[:,1:-1]

sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)
k_init = 2
k_end = 15
silueta = []
k = []
siluetaAux = []
kAux = []
semilla = []
pca=PCA(n_components=8)
pca.fit(X)
X_pca=pca.transform(X)
expl =pca.explained_variance_ratio_
mat_cov= pca.get_covariance()

for jj in range(10):
    for ii in range(k_init,k_end):
        kmeans=KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
            n_clusters=ii, n_init=20,
            random_state=jj, tol=0.0001, verbose=0)
        kAux.append(ii)
        cluster_labels = kmeans.fit_predict(X_pca)
        siluetaAux.append(silhouette_score(X_pca,cluster_labels,metric='euclidean'))
        
    maxSilueta = max(siluetaAux)
    maxIndex = siluetaAux.index(maxSilueta)
    silueta.append(maxSilueta)
    semilla.append(jj)
    
    k.append(kAux[maxIndex])

    kAux = []
    siluetaAux = []

maxSilueta = max(silueta)
maxIndex = silueta.index(maxSilueta)
kFinal = k[maxIndex]
semillaFinal = semilla[maxIndex]
print('El mejor valor de coeficiente de silueta('+str(maxSilueta)+') se obtuvo con la semilla: '+ str(semillaFinal) +' y con k ='+str(kFinal))

cluster=KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
            n_clusters=kFinal, n_init=20,
            random_state=semillaFinal, tol=0.0001, verbose=0)



    
    




