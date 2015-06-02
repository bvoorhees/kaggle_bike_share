# -*- coding: utf-8 -*-
'''
IMPORTANT: HIT F5 TO LOAD THE DATA QUICKLY AND RUN THIS WHOLE FILE
A VARIABLE CODEBOOK IS AVAILABLE IN "PROJECT WRITEUP.PDF"
'''

import pandas.stats.moments as ps
import pandas as pd

#read in the submission dataset for kaggle (has no demand information)
submit =  pd.read_csv('./raw_data/test.csv').dropna() 
submit.isnull().sum()==0 #clean!
#read in the data I'll use to fit the model. 
dat = pd.read_csv('./raw_data/train.csv').dropna() 
dat.isnull().sum()==0 #clean!

#Have to rename the count variable to "Demand" since count is a method in python
dat.rename(columns={'count':'demand'},inplace=True)

#Create list of lagged variables 
lag_vars=['weather','atemp','humidity','windspeed']

#Create dummies for certain time variables
dummy_vars=[]

#make the list of features for regressions and decision trees
feature_list=['Year','Hour','Workday','holiday'] #base features
for var in lag_vars:
  for lag in [1,2]:
    feature_list.append(str(var+"_ravg_"+str(lag)))          
    feature_list.append(str(var+"_l_"+str(lag)))
    feature_list.append(str(var+"_l_"+str(lag)+"_sq"))       

del(lag)

def FeatureEngineer(data,lagvars,dummies):  
  
  #Break out granular time variable (format 2011-01-01 05:00:00)

  data.rename(columns={'season':'Season','workingday':'Workday'}, inplace=True)
  data['datetime_unformatted']=data.datetime
  data['datetime'] = pd.to_datetime(data.datetime)
  data.set_index('datetime', inplace=True)
  data['Year'] = data.index.year
  data['Month'] = data.index.month
  data['Day'] = data.index.day
  data['Weekday'] = data.index.weekday
  data['Hour'] = data.index.hour
  data.reset_index(inplace=True)
  #make dummy variables and store in a list
  
  for var in dummies:
    dummy=pd.get_dummies(data[var],prefix=var).iloc[:,1:]
    data = pd.concat([data,dummy],axis=1, join = 'inner')  
  
  #make lag variables and store in a list (feature_list)
  for var in lagvars:
    for lag in [1,2,3,4,6,12,24,36,48]: 
      data[str(var+"_ravg_"+str(lag))]=ps.rolling_mean(data[var].shift(1),lag)
      data[str(var+"_ravg_"+str(lag))].fillna(value=data[var],inplace=True)
      data[str(var+"_l_"+str(lag))]=data[var].shift(lag)
      data[str(var+"_l_"+str(lag))].fillna(value=data[var],inplace=True)
      data[str(var+"_l_"+str(lag)+"_sq")]=data[var].shift(lag).pow(2)
      data[str(var+"_l_"+str(lag)+"_sq")].fillna(value=data[var],inplace=True)
           
  #Now make some other features (groups) that're useful for graphing#
  data['AM_PM']= data.Hour  
  data.AM_PM[data.Hour<12]=0 #AM
  data.AM_PM[data.Hour>=12]=1 #PM
  
  data['Time_of_Day']= data.Hour  
  data.Time_of_Day[(data.Hour>=2) & (data.Hour<=5)]=0 #"Very Early AM" 
  data.Time_of_Day[(data.Hour>=6) & (data.Hour<=9)]=1 #"Morning Commute"
  data.Time_of_Day[(data.Hour>=10) & (data.Hour<=13)]=2 #"Mid-morning and lunch"
  data.Time_of_Day[(data.Hour>=14) & (data.Hour<=17)]=3 #"Mid Afternoon"
  data.Time_of_Day[(data.Hour>=18) & (data.Hour<=21)]=4 #"Evening Commute and HH rush"
  data.Time_of_Day[(data.Hour==22) | (data.Hour==23)| (data.Hour==0)| (data.Hour==1)]=5 #"Late PM"
  
  #Convert "feels like temperature" to farenheit 
  data['atemp2']=data.atemp*9/5+32 
  
  data['Weather_Temp']= data.atemp    
  data.Weather_Temp[data.atemp<50]=0
  data.Weather_Temp[(data.atemp>=50) & (data.atemp<65)]=1
  data.Weather_Temp[(data.atemp>=65) & (data.atemp<80)]=2
  data.Weather_Temp[(data.atemp>=80) & (data.atemp<95)]=3
  data.Weather_Temp[(data.atemp>=95) & (data.atemp<110)]=4
  data.Weather_Temp[data.atemp>=110]=5
  
  #sort the data  
  data.set_index('datetime',inplace=True)
  data.sort_index(inplace=True)
  data.reset_index(inplace=True)
  
  del(lag, var)  
  return data
  
#run the feature engineer over training data; make dummies for some time variables  
dat = FeatureEngineer(dat,lag_vars,dummy_vars)
#run the feature engineer over training data; make dummies for some time variables
submit = FeatureEngineer(submit,lag_vars,dummy_vars)

del(lag_vars, var,dummy_vars)

