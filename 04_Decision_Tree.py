# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 15:04:36 2015

@author: beevo
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV

#import the data
runfile('./01_Import_and_clean.py')

#split the data into test and training data sets
train, test=dat.iloc[:int(len(dat)*.8),:], dat.iloc[int(len(dat)*.8):,:]

###CASUAL MODEL###
# Conduct a grid search for the best tree depth

ctree = tree.DecisionTreeRegressor(random_state=1,min_samples_split=20,min_samples_leaf=20,max_depth=15,max_features=27)
depth_range = range(10,220,10)
param_grid = dict(n_estimators=depth_range)
grid = GridSearchCV(ctree, param_grid, cv=5, scoring='mean_squared_error')
grid.fit(dat[feature_list], dat.casual)

# Check out the scores of the grid search
grid_mean_scores = [np.mean(np.sqrt(-result[1])) for result in grid.grid_scores_]

# Plot the results of the grid search
plt.figure()
plt.plot(depth_range, grid_mean_scores)
plt.hold(True)
plt.grid(True)
plt.plot(grid.best_params_['max_features'], np.sqrt(-grid.best_score_), 'ro', markersize=12, markeredgewidth=1.5,
         markerfacecolor='None', markeredgecolor='r')

##FIT THE MODEL         

cas_tree=tree.DecisionTreeRegressor(random_state=1,max_depth=15, min_samples_split=20,min_samples_leaf=20)
cas_tree.fit(dat[feature_list],dat.casual)
cas_scores=cross_val_score(cas_tree, dat[feature_list], dat.casual, cv=10, scoring='mean_squared_error')
np.mean(np.sqrt(-cas_scores))

dat['cas_tpreds']=cas_tree.predict(dat[feature_list])
cas_feat_importance=pd.DataFrame(zip(cas_tree.feature_importances_,feature_list), columns=['gini_importance','feature'])
cas_feat_importance.sort_index(by='gini_importance', inplace=True,ascending=False)
cas_feat_importance.head(10)

#show predicted vs. actual plot
p1=plt.scatter(dat.casual,dat.cas_tpreds) 
x=range(0,dat.casual.max())
y=range(0,dat.casual.max())
p2=plt.plot(x,y,c='red')
plt.xlim(0,max(test.casual))
plt.xlabel("Actual")
plt.ylim(0,max(test.cas_tpreds))
plt.ylabel("Predicted")
plt.show()

###REGISTERED MODEL###

# Conduct a grid search for the best tree depth
rtree = tree.DecisionTreeRegressor(random_state=1, min_samples_split=20,min_samples_leaf=20,max_depth=14)
depth_range = range(1, len(feature_list)+1)
param_grid = dict(n_estimators=depth_range)
grid = GridSearchCV(rtree, param_grid, cv=5, scoring='mean_squared_error')
grid.fit(dat[feature_list], dat.registered)

# Check out the scores of the grid search
grid_mean_scores = [np.mean(np.sqrt(-result[1])) for result in grid.grid_scores_]

# Plot the results of the grid search
plt.figure()
plt.plot(depth_range, grid_mean_scores)
plt.hold(True)
plt.grid(True)
plt.plot(grid.best_params_['max_features'], np.sqrt(-grid.best_score_), 'ro', markersize=12, markeredgewidth=1.5,
         markerfacecolor='None', markeredgecolor='r')

##Fitting the model##

reg_tree=tree.DecisionTreeRegressor(random_state=1,max_depth=12,min_samples_split=20,min_samples_leaf=20)
reg_tree.fit(train[feature_list],train.registered)
reg_scores=cross_val_score(reg_tree, train[feature_list], train.registered, cv=10, scoring='mean_squared_error')
np.mean(np.sqrt(-reg_scores))

test['reg_tpreds']=reg_tree.predict(test[feature_list])
reg_feat_importance=pd.DataFrame(zip(reg_tree.feature_importances_,feature_list),columns=['gini_importance','feature'])
reg_feat_importance.sort_index(by='gini_importance',inplace=True, ascending=False)
reg_feat_importance.head(10)

#show predicted vs. actual plot
p1=plt.scatter(test.registered,test.reg_tpreds) 
x=range(0,test.registered.max())
y=range(0,test.registered.max())
p2=plt.plot(x,y,c='red')
plt.xlim(0,max(test.registered))
plt.xlabel("Actual")
plt.ylim(0,max(test.reg_tpreds))
plt.ylabel("Predicted")
plt.show()


###MAKE SUBMISSION TO KAGGLE###
cas_preds=cas_tree.predict(submit[feature_list])
reg_preds=reg_tree.predict(submit[feature_list])

count = [int(round(i+j)) for i,j in zip(reg_preds, cas_preds)]
df_submission = pd.DataFrame(count, submit.datetime, columns = ['count'])
pd.DataFrame.to_csv(df_submission ,'./submission_files/decision_tree.csv')