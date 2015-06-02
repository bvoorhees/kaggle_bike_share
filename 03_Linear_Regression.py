# -*- coding: utf-8 -*-
'''
IMPORTANT: HIT F5 IN 01_Import_and_Clean.Py TO LOAD THE DATA QUICKLY FOR THIS FILE
A VARIABLE CODEBOOK IS AVAILABLE IN "PROJECT WRITEUP.PDF"
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score

dat=dat[dat.isnull().sum(axis=1) == 0]
submit=submit[submit.isnull().sum(axis=1) == 0]


#sklearn spits out dataframes with numbers that trigger errors, doing a manual split below

train, test=dat.iloc[:int(len(dat)*.8),:], dat.iloc[int(len(dat)*.8):,:]

#Here are all of the explanitory variables

slm=LinearRegression()

#loop over all the explanitory variables to regress each one individually on registered user demand
reg_scores= [np.mean(np.sqrt(-cross_val_score(slm, 
                                              dat[var][:,np.newaxis], 
                                              dat['registered'], 
                                              cv=10, 
                                              scoring='mean_squared_error'))) 
                                              for var in feature_list]
#loop over all the explanitory variables to regress each one individually on casual user demand
cas_scores= [np.mean(np.sqrt(-cross_val_score(slm, 
                                              dat[var][:,np.newaxis], 
                                              dat['casual'], 
                                              cv=10, 
                                              scoring='mean_squared_error')))
                                              for var in feature_list]
#put the RMSEs of regressions above into dataframes 
reg_dict={'RMSE':reg_scores ,'variable':feature_list}
cas_dict ={'RMSE':cas_scores ,'variable':feature_list} 

cas_scores=pd.DataFrame(cas_dict)
reg_scores=pd.DataFrame(reg_dict)

#Now rank the explanitory variables by lowest to highest RMSLE 
cas_scores.sort_index(by='RMSE', inplace=True)
reg_scores.sort_index(by='RMSE', inplace=True)

#looks like I'll start messign with a multi-variable linear model using the results above
#I'm not going to use temperature variables twice (since they're related), 
#note weekday variables are all relative to Monday


#The RMSLE's don't much more stellar right now (relative to hourly averages), but I'll make predictions

'''
CASUAL MODEL
'''
casmodelX=['temp','Workday','Hour','humidity','Weekday']
mlmc=LinearRegression()
mlmc.fit(train[casmodelX],train.casual)
test.casual_fit=mlmc.predict(test[casmodelX])

#plot fits and actuals
p1=plt.scatter(test.casual,test.casual_fit) 
x=range(0,300)
y=range(0,300)
p2=plt.plot(x,y,c='red')
plt.xlim(0,max(test.casual))
plt.xlabel("Actual")
plt.ylim(0,max(test.casual_fit))
plt.ylabel("Predicted")
plt.savefig('./graphs/cas_linear.png',bbox_inches='tight', transparent=True,dpi=150) 
plt.show()


#show me scores of the casual model
cas_score=cross_val_score(mlmc, train[casmodelX], train['casual'], cv=10,
                          scoring='mean_squared_error')
np.mean(np.sqrt(-cas_score))


'''
REGISTERED MODEL
'''
regmodelX=['Hour','temp','Year','humidity','Workday']
mlmr=LinearRegression()
mlmr.fit(train[regmodelX],train.registered)
test.registered_fit=mlmr.predict(test[regmodelX])

#plot fits and actuals
p1=plt.scatter(test.registered,test.registered_fit) 
x=range(0,300)
y=range(0,300)
p2=plt.plot(x,y,c='red')
plt.xlim(0,max(test.registered))
plt.xlabel("Actual")
plt.ylim(0,max(test.registered_fit))
plt.ylabel("Predicted")
plt.savefig('./graphs/reg_linear.png',bbox_inches='tight', transparent=True,dpi=150) 
plt.show()

#show me scores of the registered model
reg_score=cross_val_score(mlmr, train[regmodelX], train['registered'], cv=10,
                          scoring='mean_squared_error')
np.mean(np.sqrt(-reg_score))


'''
 MAKE SUBMISSION TO KAGGLE
'''

#run demand modelS on kaggle submission data to make predicitons
cas_preds = mlmc.predict(submit[casmodelX])
reg_preds = mlmr.predict(submit[regmodelX])

#now zip the casual and registered predicted data for submission together and 
#shoot out CSV for submission
count = [int(round(i+j)) for i,j in zip(reg_preds, cas_preds)]

count = [0 if i <0 else i for i in count]

df_submission = pd.DataFrame(count, submit.datetime, columns = ['count'])

pd.DataFrame.to_csv(df_submission ,'./submission_files/linear_model.csv')

'''
USE STATSMODELS TO BETTER UNDERSTAND COEFFICIENTS
'''
import statsmodels.formula.api as smf


#construct a string for the casual demand formula using casmodelX list and regmodelX list
casmodel_formula="casual~"

count=1
for var in casmodelX:
  if count<len(casmodelX):    
    casmodel_formula= casmodel_formula+var+" + "
  else:
    casmodel_formula= casmodel_formula+var
  count=count+1

#construct a string for the formula
regmodel_formula="registered~"

count=1
for var in regmodelX:
  if count<len(regmodelX):    
    regmodel_formula= regmodel_formula+var+" + "
  else:
    regmodel_formula= regmodel_formula+var
  count=count+1
#Registered model output
lmr = smf.ols(formula=regmodel_formula, data=train).fit()
lmr.summary()
#Casual Model output
lmc = smf.ols(formula=casmodel_formula, data=train).fit()
lmc.summary()

