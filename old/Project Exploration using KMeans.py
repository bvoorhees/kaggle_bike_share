# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 21:38:25 2015

@author: beevo
"""

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.tools.plotting import parallel_coordinates

dat = train[train['Year']==2012][:]

demand=pd.DataFrame((dat.demand-dat.demand.mean())/dat.demand.std())
registered=pd.DataFrame((dat.registered-dat.registered.mean())/dat.registered.std())
casual=pd.DataFrame((dat.casual-dat.casual.mean())/dat.casual.std())

atemp=pd.DataFrame((dat.atemp-dat.atemp.mean())/dat.atemp.std())
hour=pd.DataFrame((dat.Hour-dat.Hour.mean())/dat.Hour.std())

workday=dat.Workday

c=pd.concat([demand,hour, workday,atemp],axis=1)

k_rng = range(2,16)
est_demand = [KMeans(n_clusters = k).fit(c) for k in k_rng]

#================================
# Option 1: Silhouette Coefficient
# Generally want SC to be closer to 1, while also minimizing k

from sklearn import metrics
silhouette_score = [metrics.silhouette_score(c, e.labels_, metric='euclidean') for e in est_demand]
# Plot the results

plt.figure(figsize=(7, 8))
plt.subplot(211)
plt.title('Using the elbow method to inform k choice')
plt.plot(k_rng, silhouette_score, 'b*-')
plt.xlim([1,14])
plt.grid(True)
plt.ylabel('Silhouette Coefficient')

#Will choose 13...

est = KMeans(n_clusters=13, init='random')
est.fit(c)
y_kmeans = est.predict(c)


# Create at least one data visualization (e.g., Scatter plot grid, 3d plot, parallel coordinates)

c['est'] =y_kmeans

plt.figure()
parallel_coordinates(c,'est')#, colors=('#FF0054','#FBD039'))
 
