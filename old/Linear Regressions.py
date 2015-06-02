# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 20:58:27 2015

@author: beevo
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split

training=train
#training.isnull().sum()==0 #clean!

'''
train, test = train_test_split(training, test_size=0.3, random_state=1)
train = pd.DataFrame(data=train, columns=training.columns)
test = pd.DataFrame(data=test, columns=training.columns)
'''

Xlist=['Season', 'Weekday', 'Workday','Time_of_Day', 'Month','Day','Hour', 'Weather_Temp']

def reg_plot(Xes,y):
  count=1  
  subplot = 221   
  for x in Xes:  
    slm = LinearRegression()
    slm.fit(train[x][:,np.newaxis], train[y]) #requires x data to be two dimensiona;- np.newaxis makes this possible
    
    # Evaluate the output
  
    y_min=slm.intercept_+ slm.coef_*train[x].min()
    y_max=slm.intercept_+ slm.coef_*train[x].max()
    
    plt.subplot(subplot)    
    plt.plot([train[x].min(), train[x].max()], [y_min, y_max],color='r')
    plt.scatter(x=train[x],y=train[y])

    plt.yticks(np.arange(0,round(train[y].max()+10,-1),10))
          
    plt.ylabel(y)
    plt.xlabel(x)
    slm2=LinearRegression()
    scores = cross_val_score(slm2, training[x][:,np.newaxis], training[y], cv=10, scoring='mean_squared_error')
    print(y)+"~"+x+" Score: "+str(np.mean(np.sqrt(-scores)))
    print("Beta: "+str(slm.coef_[0]))     
    if count==4:
      subplot=subplot+7
      count=1
    else:
      subplot=subplot+1      
      count=count+1

reg_plot(Xlist,'casual')
reg_plot(Xlist,'registered')