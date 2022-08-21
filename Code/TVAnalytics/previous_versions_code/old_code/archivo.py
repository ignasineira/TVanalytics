import os
import pandas as pd
import numpy as np
import datetime
from itertools import combinations , product

np.random.seed(seed=1234)
date = pd.date_range(end=datetime.date.today(), periods=50)
print(date)
datelist = date.tolist()
date_int = [int(time.strftime('%Y%m%d'))for time in datelist] #date list to int
dict_time_block = {'1': '6:00', '2': '6:30', '3': '7:00', '4': '7:30','5': '8:00', '6': '8:30', '7': '9:00',
                   '8': '9:30', '9': '10:00', '10': '10:30', '11': '11:00', '12': '11:30', '13': '12:00', '14': '12:30',
                   '15': '13:00', '16': '13:30', '17': '14:00', '18': '14:30', '19': '15:00', '20': '15:30',
                   '21': '11:00', '22': '11:30', '23': '12:00', '24': '12:30', '25': '13:00', '26': '13:30',
                   '27': '14:00', '28': '14:30', '29': '15:00', '30': '15:30', '31': '16:00', '32': '16:30',
                   '33': '17:00', '34': '17:30','35': '18:00', '36': '18:30', '37': '19:00', '38': '19:30',
                   '39': '20:00', '40': '20:30', '41': '01:00', '40': '01:30', '41': '02:00', '42': '2:30'
                   }
time_block_keys = dict_time_block.keys()
time_block_keys=[key for key in time_block_keys] #key list to int()
tuplas_date_tb = list(product(datelist, time_block_keys))
df = pd.DataFrame(tuplas_date_tb, columns=["start_date", "time_block"])
list_rating_tb=[1,1,2,2,3,3,4,4,5,5,5,5,5,5,4,4,4,4,2,2,2,2,2,2,3,3,1,1,1,2,2,4,4,4,4,4,7,8,9,2,2,1]
sigma=10
df["rating"]=0
for i in range(len(time_block_keys)):
    df.loc[df.time_block==time_block_keys[i],"rating"] = list_rating_tb[i]

df["rating"] = [ np.maximum(df.loc[k,"rating"]+np.random.randn(), 0) for k in df.index]
df['day']=pd.DatetimeIndex(df['start_date']).day
df['month']=pd.DatetimeIndex(df['start_date']).month
df['week_day']=df['start_date'].dt.weekday
days = {0:'Mon',1:'Tues',2:'Weds',3:'Thurs',4:'Fri',5:'Sat',6:'Sun'}

print(df[df.time_block=='1'].tail(40))
semanas=4
df['lag_1']=df.groupby(['week_day', 'time_block'])["rating"].diff(semanas*1)
df['lag_2']=df.groupby(['week_day', 'time_block'])["rating"].diff(semanas*2)
df['lag_3']=df.groupby(['week_day', 'time_block'])["rating"].diff(semanas*3)
df['lag_4']=df.groupby(['week_day', 'time_block'])["rating"].diff(semanas*4)
df['lag_5']=df.groupby(['week_day', 'time_block'])["rating"].diff(semanas*5)
df['lag_6']=df.groupby(['week_day', 'time_block'])["rating"].diff(semanas*6)
df['lag_7']=df.groupby(['week_day', 'time_block'])["rating"].diff(semanas*7)
df['lag_8']=df.groupby(['week_day', 'time_block'])["rating"].diff(semanas*8)
df['lag_9']=df.groupby(['week_day', 'time_block'])["rating"].diff(semanas*9)
df['lag_10']=df.groupby(['week_day', 'time_block'])["rating"].diff(semanas*10)
df['lag_11']=df.groupby(['week_day', 'time_block'])["rating"].diff(semanas*11)
df['lag_12']=df.groupby(['week_day', 'time_block'])["rating"].diff(semanas*12)
#print(df.head())
print(df[(df.time_block=='1')& (df.week_day==6)].tail(40))

print (df)

