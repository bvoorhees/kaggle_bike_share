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
from sklearn.ensemble import RandomForestRegressor as rf

#import the data
runfile('./01_Import_and_clean.py')

#split the data into test and training data sets
train, test=dat.iloc[:int(len(dat)*.8),:], dat.iloc[int(len(dat)*.8):,:]

###CASUAL MODEL###
  
  #Tune a bit (did tree depth already in decision tree file)#
  cas_forest=rf(criterion='mse',
                max_depth=15, 
                min_samples_split=5,
                min_samples_leaf=1,
                max_features=27,
                max_leaf_nodes=None,
                bootstrap=True,
                oob_score=True,
                n_jobs=-1,
                random_state=None, 
                verbose=0)
  
  depth_range = range(10,220,10)
  param_grid = dict(n_estimators=depth_range)
  grid = GridSearchCV(cas_forest, param_grid, cv=5, scoring='mean_squared_error')
  grid.fit(dat[feature_list], dat.casual)
  
  # Check out the scores of the grid search
  grid_mean_scores = [np.mean(np.sqrt(-result[1])) for result in grid.grid_scores_]
  
  # Plot the results of the grid search
  plt.figure()
  plt.plot(depth_range, grid_mean_scores)
  plt.hold(True)
  plt.grid(True)
  plt.plot(grid.best_params_['n_estimators'], np.sqrt(-grid.best_score_), 'ro', markersize=12, markeredgewidth=1.5,
           markerfacecolor='None', markeredgecolor='r')
  
  ##now fit
  cas_forest=rf(n_estimators=1000,
                criterion='mse',
                max_depth=15, 
                min_samples_split=5,
                min_samples_leaf=1,
                max_features=27,
                max_leaf_nodes=None,
                bootstrap=True,
                oob_score=True,
                n_jobs=-1,
                random_state=None, 
                verbose=0)
  #scores              
  cas_scores=cross_val_score(cas_forest, dat[feature_list], dat.casual, cv=10, scoring='mean_squared_error')
  np.mean(np.sqrt(-cas_scores))
  #predictions on test set
  cas_forest.fit(train[feature_list],train.casual)
  test['cas_tpreds']=cas_forest.predict(test[feature_list])
  
  # Feature Importances
  cas_feat_importance=pd.DataFrame(zip(cas_forest.feature_importances_,feature_list), columns=['gini_importance','feature'])
  cas_feat_importance.sort_index(by='gini_importance', inplace=True,ascending=False)
  cas_feat_importance.head(10)
  #plots actuals vs. predicted
  p1=plt.scatter(test.casual,test.cas_tpreds) 
  x=range(0,test.casual.max())
  y=range(0,test.casual.max())
  p2=plt.plot(x,y,c='red')
  plt.xlim(0,max(test.casual))
  plt.xlabel("Actual")
  plt.ylim(0,max(test.cas_tpreds))
  plt.ylabel("Predicted")
  plt.savefig('./graph/cas_rf.png',bbox_inches='tight', transparent=True,dpi=150) 
  plt.show()

###REGISTERED MODEL###

  #Tune a bit (did tree depth already in decision tree file)#
  reg_forest=rf(criterion='mse',
                max_depth=15, 
                min_samples_split=5,
                min_samples_leaf=1,
                max_features=27,
                max_leaf_nodes=None,
                bootstrap=True,
                oob_score=True,
                n_jobs=-1,
                random_state=None, 
                verbose=0)
  
  depth_range = range(10,220,10)
  param_grid = dict(n_estimators=depth_range)
  grid = GridSearchCV(reg_forest, param_grid, cv=5, scoring='mean_squared_error')
  grid.fit(dat[feature_list], dat.registered)
  
  # Check out the scores of the grid search
  grid_mean_scores = [np.mean(np.sqrt(-result[1])) for result in grid.grid_scores_]
  
  # Plot the results of the grid search
  plt.figure()
  plt.plot(depth_range, grid_mean_scores)
  plt.hold(True)
  plt.grid(True)
  plt.plot(grid.best_params_['n_estimators'], np.sqrt(-grid.best_score_), 'ro', markersize=12, markeredgewidth=1.5,
           markerfacecolor='None', markeredgecolor='r')
  
  #now Fit
  
  reg_forest=rf(n_estimators=180,
                criterion='mse',
                max_depth=15, 
                min_samples_split=5,
                min_samples_leaf=1,
                max_features=26,
                max_leaf_nodes=None,
                bootstrap=True,
                oob_score=True,
                n_jobs=-1,
                random_state=None, 
                verbose=0)
  #scores              
  reg_scores=cross_val_score(reg_forest, dat[feature_list], dat.registered, cv=10, scoring='mean_squared_error')
  np.mean(np.sqrt(-reg_scores))
  #predictions on test set
  reg_forest.fit(train[feature_list],train.registered)
  test['reg_tpreds']=reg_forest.predict(test[feature_list])
  #feature importances
  reg_feat_importance=pd.DataFrame(zip(reg_forest.feature_importances_,feature_list),columns=['gini_importance','feature'])
  reg_feat_importance.sort_index(by='gini_importance',inplace=True, ascending=False)
  reg_feat_importance.head(10)
  #plots actuals vs. predicted
  p1=plt.scatter(test.registered,test.reg_tpreds) 
  x=range(0,test.registered.max())
  y=range(0,test.registered.max())
  p2=plt.plot(x,y,c='red')
  plt.xlim(0,max(test.registered))
  plt.xlabel("Actual")
  plt.ylim(0,max(test.reg_tpreds))
  plt.ylabel("Predicted")
  plt.savefig('./graph/reg_rf.png',bbox_inches='tight', transparent=True,dpi=150) 
  plt.show()


###MAKE SUBMISSION TO KAGGLE###
cas_forest.fit(dat[feature_list],dat.casual)
cas_preds=cas_forest.predict(dat[feature_list])
reg_forest.fit(dat[feature_list],dat.registered)
reg_preds=reg_forest.predict(submit[feature_list])

count = [int(round(i+j)) for i,j in zip(reg_preds, cas_preds)]
df_submission = pd.DataFrame(count, submit.datetime, columns = ['count'])
pd.DataFrame.to_csv(df_submission ,'./submission_files/random_forest.csv')