# -*- coding: utf-8 -*-
'''
IMPORTANT: HIT F5 IN 01_Import_and_Clean.Py TO LOAD THE DATA QUICKLY FOR THIS FILE
HIT F5 IN THIS FILE TO LOAD THE GRAPH PLOTTER AND TYPE "dographs()" TO PLOT GRAPHS
A VARIABLE CODEBOOK IS AVAILABLE IN "PROJECT WRITEUP.PDF"
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#make a list of the graphs I want to see
graphs = ['Season', 'Weekday', 'Workday','Time_of_Day', 'Year','Month','Day','Hour', 'Weather_Temp']

def kagglechart(frequency):
  
  #first collapse data into hourly averages for registered (reg) and casual (cas) demand
  reg=dat.groupby(['Year',frequency]).registered.sum()/dat.groupby(['Year',frequency]).registered.count()  
  
  cas=dat.groupby(['Year',frequency]).casual.sum()/dat.groupby(['Year',frequency]).casual.count()
 
 #now make some labels and graph parameters
  Season=('Winter','Spring', 'Summer', 'Fall')
  Weekday=('Mon.','Tue.','Wen.','Thu.','Fri.','Sat.','Sun.')
  Workday=("Playday","Workday")
  Time_of_Day=("Early AM","AM Commute", "Mid AM+Lunch","Mid PM", "PM Commute","Late PM")  
  Year = tuple(set(dat.Year))
  Month = tuple(set(dat.Month))
  Day = tuple(set(dat.Day))
  Hour = tuple(set(dat.Hour))
  Weather_Temp=("<50","50-65","65-80","80-95","95-110",">110")
  #put all the label types into a list  
  labels=[Season,Weekday,Year,Month,Day,Workday,Hour,Time_of_Day,Weather_Temp]
  
  #put all those labels into a dictionary   
  dict={'Season':0,'Weekday':1,'Year':2,'Month':3,'Day':4,'Workday':5,'Hour':6,'Time_of_Day':7,'Weather_Temp':8}   
  
  #specify the parameters of the stacked barchart graph
  width = 0.35       
  
  p1 = plt.bar(np.arange(1,len(reg)+1),reg, width, color='r')
  p2 = plt.bar(np.arange(1,len(cas)+1),cas, width, color='b', bottom=reg)
  
  #axis labels
  plt.ylabel('Demand')

  if frequency!='Year':  
    plt.xlabel('2011                                       2012')
  
  #graph title
  plt.title('Average Hourly Demand by '+frequency)
  
  #axis tickmarks
  plt.xticks(np.arange(1,len(reg)+1)+width/2.,labels[dict[frequency]]+labels[dict[frequency]],rotation=90)
  
  #legend
  plt.legend( (p1[0], p2[0]), ('Registered Renters', 'Casual Renters'),loc=0 )
  #save out and display  
  plt.savefig('./graphs/Average Hourly Demand by '+frequency+'.png',bbox_inches='tight', transparent=True,dpi=150) 
  plt.show()

#make a quick function to run if I decide to graph stuff on the fly
#hit F5 in python, then in the counsol type dographs()
def dographs():
  for g in graphs:
    kagglechart(g)
    

