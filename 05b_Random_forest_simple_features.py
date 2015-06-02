# -*- coding: utf-8 -*-
"""
Created on Thu May  7 22:39:31 2015

@author: beevo
"""
import pylab as pl
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from datetime import datetime
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
 


np.set_printoptions(threshold=np.nan)
 
#engineer features 
def transform(df):
  i = 0
  for timestamp in df['datetime']:
    i += 1
    date_object = datetime.strptime(timestamp.split()[0], '%Y-%m-%d')
    time = timestamp.split()[1][:2]
    date = datetime.date(date_object).weekday()
    df.loc[i-1, 'date'] = date
    df.loc[i-1, 'time'] = time   
  return df

df_train = pd.read_csv('./raw_data/train.csv')
df_test = pd.read_csv('./raw_data/test.csv')
train, test = transform(df_train), transform(df_test)

cols = ['date','time', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed']

### 1 Tune number of estimators for casual model###
    rf = RandomForestRegressor(oob_score=True,n_jobs=-1)
    
    depth_range = range(100,2200,200)
    param_grid = dict(n_estimators=depth_range)
    grid = GridSearchCV(rf, param_grid, cv=5, scoring='mean_squared_error')
    grid.fit(train[cols], train.casual)
    grid_mean_scores = [np.mean(np.sqrt(-result[1])) for result in grid.grid_scores_]
    
    # Plot the results of the grid search
    plt.figure()
    plt.plot(depth_range, grid_mean_scores)
    plt.hold(True)
    plt.grid(True)
    plt.plot(grid.best_params_['n_estimators'], np.sqrt(-grid.best_score_), 'ro', markersize=12, markeredgewidth=1.5,
             markerfacecolor='None', markeredgecolor='r')

### 2 Tune number of splits for casual model###
    rf = RandomForestRegressor(n_estimators=900, oob_score=True,n_jobs=-1)
    
    depth_range = range(1,len(cols))
    param_grid = dict(min_samples_split=depth_range)
    grid = GridSearchCV(rf, param_grid, cv=5, scoring='mean_squared_error')
    grid.fit(train[cols], train.casual)
    grid_mean_scores = [np.mean(np.sqrt(-result[1])) for result in grid.grid_scores_]
    
    # Plot the results of the grid search
    plt.figure()
    plt.plot(depth_range, grid_mean_scores)
    plt.hold(True)
    plt.grid(True)
    plt.plot(grid.best_params_['min_samples_split'], np.sqrt(-grid.best_score_), 'ro', markersize=12, markeredgewidth=1.5,
             markerfacecolor='None', markeredgecolor='r')

### 3 Tune number of splits for casual model###
    rf = RandomForestRegressor(n_estimators=900, min_samples_split=9,oob_score=True,n_jobs=-1)
    
    depth_range = range(1,30,2)
    param_grid = dict(max_depth=depth_range)
    grid = GridSearchCV(rf, param_grid, cv=5, scoring='mean_squared_error')
    grid.fit(train[cols], train.casual)
    grid_mean_scores = [np.mean(np.sqrt(-result[1])) for result in grid.grid_scores_]
    
    # Plot the results of the grid search
    plt.figure()
    plt.plot(depth_range, grid_mean_scores)
    plt.hold(True)
    plt.grid(True)
    plt.plot(grid.best_params_['max_depth'], np.sqrt(-grid.best_score_), 'ro', markersize=12, markeredgewidth=1.5,
             markerfacecolor='None', markeredgecolor='r')


rf = RandomForestRegressor(n_estimators=900, min_samples_split=9,max_depth=7 ,oob_score=True,n_jobs=-1)

casual = rf.fit(train[cols], train.casual)
print casual.feature_importances_

predict_casual = rf.predict(test[cols])

### 1 Tune number of estimators for registered model###
    rf = RandomForestRegressor(oob_score=True,n_jobs=-1)
    
    depth_range = range(100,2200,200)
    param_grid = dict(n_estimators=depth_range)
    grid = GridSearchCV(rf, param_grid, cv=5, scoring='mean_squared_error')
    grid.fit(train[cols], train.registered)
    grid_mean_scores = [np.mean(np.sqrt(-result[1])) for result in grid.grid_scores_]
    
    # Plot the results of the grid search
    plt.figure()
    plt.plot(depth_range, grid_mean_scores)
    plt.hold(True)
    plt.grid(True)
    plt.plot(grid.best_params_['n_estimators'], np.sqrt(-grid.best_score_), 'ro', markersize=12, markeredgewidth=1.5,
             markerfacecolor='None', markeredgecolor='r')

### 2 Tune number of splits for registered model###
    rf = RandomForestRegressor(n_estimators=900, oob_score=True,n_jobs=-1)
    
    depth_range = range(1,len(cols))
    param_grid = dict(min_samples_split=depth_range)
    grid = GridSearchCV(rf, param_grid, cv=5, scoring='mean_squared_error')
    grid.fit(train[cols], train.registered)
    grid_mean_scores = [np.mean(np.sqrt(-result[1])) for result in grid.grid_scores_]
    
    # Plot the results of the grid search
    plt.figure()
    plt.plot(depth_range, grid_mean_scores)
    plt.hold(True)
    plt.grid(True)
    plt.plot(grid.best_params_['min_samples_split'], np.sqrt(-grid.best_score_), 'ro', markersize=12, markeredgewidth=1.5,
             markerfacecolor='None', markeredgecolor='r')

### 3 Tune number of splits for registered model###
    rf = RandomForestRegressor(n_estimators=900, min_samples_split=4,oob_score=True,n_jobs=-1)
    
    depth_range = range(1,30,2)
    param_grid = dict(max_depth=depth_range)
    grid = GridSearchCV(rf, param_grid, cv=5, scoring='mean_squared_error')
    grid.fit(train[cols], train.registered)
    grid_mean_scores = [np.mean(np.sqrt(-result[1])) for result in grid.grid_scores_]
    
    # Plot the results of the grid search
    plt.figure()
    plt.plot(depth_range, grid_mean_scores)
    plt.hold(True)
    plt.grid(True)
    plt.plot(grid.best_params_['max_depth'], np.sqrt(-grid.best_score_), 'ro', markersize=12, markeredgewidth=1.5,
             markerfacecolor='None', markeredgecolor='r')


rf = RandomForestRegressor(n_estimators=900, min_samples_split=4,max_depth=13 ,oob_score=True,n_jobs=-1)
registered = rf.fit(train[cols], train.registered)
print registered.feature_importances_
predict_registered = rf.predict(test[cols])
 
count = [int(round(i+j)) for i,j in zip(predict_casual, predict_registered)]
 
# submission

df_submission = pd.DataFrame(count, test.datetime, columns = ['count'])
pd.DataFrame.to_csv(df_submission ,'./submission_files/randomforest_simple_features.csv')
