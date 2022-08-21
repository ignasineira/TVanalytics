# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 16:55:22 2021

@author: ignasi
"""


import os
import sys


import pandas as pd

module_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)

import TVAnalytics as TV
# Set your local data directory.
path = 'C:/Users/ignasi/Desktop/TVAnalytics'

os.chdir(path)

df_X=pd.read_csv('input/X_test.csv',header=None)
aux_dict={}
for i, (group_name, group_df) in enumerate(df_X.groupby([1])):
    print(group_name)
    print((group_df.isna().sum()==len(group_df)).sum())
    aux_dict[group_name]=(group_df.isna().sum()==len(group_df)).values


lista_columns = []
with open("input\columns_predict.txt", "r") as f:
    for line in f:
        lista_columns.append(line.strip())
columns_name = pd.read_csv('input\columns_name_final.csv', sep=',').columns.tolist()

na_columns=pd.DataFrame(aux_dict)
na_columns['name_columns']=columns_name

na_columns.loc[na_columns['name_columns'].isin(lista_columns),:-1]=True


cols = na_columns.columns.tolist()
cols = cols[-1:] + ['canal_'+str(i) for i in cols[:-1]]


cols = na_columns.columns.tolist()
cols1 = cols[-1:] + ['canal_'+str(i) for i in cols[:-1]]
cols = cols[-1:] + cols[:-1]
na_columns = na_columns[cols]
na_columns.columns=cols1


na_columns.to_csv('na_columns.csv', index=False)
