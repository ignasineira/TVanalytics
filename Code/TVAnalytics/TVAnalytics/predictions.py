# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 19:14:36 2021

@author: ignasi
"""

### 0.1 import package
import warnings
warnings.filterwarnings('ignore')

# Load modules.
import os
import sys
import json
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#import matplotlib.pyplot as plt
from datetime import timedelta
import datetime

from timeit import default_timer as timer
#from sklearn import linear_model, metrics, model_selection
#from sklearn.multioutput import MultiOutputRegressor
#from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import multiprocessing

#local functions
from .data_processing import generate_data
from .tunning_param import MAPE
from .plot_importance_columns import plot_importance_columns_last_week


def pred_for_last_week_data(start_year=2015,start_month=1):
    """
    This function take: 
        - best tuning parameter for each tuple (segment ,week)
        - X train 
        - the week you want predict ( generally last week of data)

    output: 
        predict the next six week of rating 
    """
    #list of columns predict
    inicio=timer()
    lista_columns = []
    with open("input\columns_predict.txt", "r") as f:
        for line in f:
            lista_columns.append(line.strip())
    df_final,list_programa=generate_data(start_year,start_month,tunning=False,for_pred=True)
    fin= timer()-inicio
    print('Data ready! time(s): ',round(fin,2))
    lista_columns=[s+'_5' for s in lista_columns]
    df_X = df_final[df_final.columns.difference(lista_columns)].copy()
    df_Y = df_final[lista_columns[::2]].copy()

    #split data
    list_tuple_week = list(set((i, j) for i, j in zip(df_final.año, df_final.semana_año)))
    list_tuple_week.sort()

    list_index = []
    for tuplas in list_tuple_week:
        list_index.append(df_final[(df_final.año == tuplas[0]) & (df_final.semana_año == tuplas[1])].index[-1])
        
    
    year_train=df_final.año.max()
    week_train=df_final[df_final.año == year_train].semana_año.max()-1
    #week_train=df_final[df_final.año == year_train].semana_año.max()-1
    if df_final[df_final.año == year_train].semana_año.max()==1:
        year_train-=1
        week_train=df_final[df_final.año == year_train].semana_año.max()
    
    index_year_week_train = df_final[(df_final.año == year_train) & (df_final.semana_año == week_train)].index[-1]

    #load best parameters
    best_parameters=[]
    dic_aux={}
    for t in range(1,7):
        for i in range(2):
            best_parameters.append(json.load(open('save_model/tunning_parameters/best_parameter_t_'+str(t)+'_target_'+str(i)+'.json', 'r')))
            dic_aux[len(best_parameters)-1]=(t,i)
            
    print('Time to train model!')
    inicio=timer()

    key_list = list(dic_aux.keys())
    val_list = list(dic_aux.values())
    
    #call columns
    dict_colums=json.load(open('input/columns_train_models/columns_train_models.json', 'r'))
    dict_colums= {int(k):v for k,v in dict_colums.items()}
    
    """
    list_columns_always_use=[col for col in df_X.columns if '_t_' not in col]
    list_columns_always_use_location=[df_X.columns.get_loc(c) for c in list_columns_always_use ]
    """
    for t in range(1, 7):
    #for t in range(2, 7):
        print("Time step predict: ", t )
        print('**--------------------------------------------**')
        print(' ')
        # Run sliding window regression models
        for target in range(2):    
            print("Target :", target )
            position = val_list.index((t,target))
            # Define test data
            #X_test = df_X.iloc[ index_year_week_train+ 1:, :]
            Y_test = df_Y.iloc[ index_year_week_train + 1:, key_list[position]]
            Yhat_test = np.zeros_like(Y_test)
            Yhat_test_all_XGB = np.zeros((Yhat_test.shape[0], 1))
            
            i=list_index.index(index_year_week_train)-(t-1)
            j=list_index.index(index_year_week_train)
            print("Training for test point:", list_tuple_week[i])
            print("Test point:", list_tuple_week[j + 1])
            
            list_columns_always_use_t_location=[df_X.columns.get_loc(c) for c in dict_colums[position]]
            """
            list_columns_use_t=[col for col in df_X.columns if '_t_'+str(t)+'_' in col]
            list_columns_always_use_t_location=[df_X.columns.get_loc(c) for c in list_columns_use_t ]
            list_columns_always_use_t_location.extend(list_columns_always_use_location)
            """
            lo = list_index[i] + 1
            lo_test = list_index[j] + 1
            hi = list_index[j + 1] + 1
            X_train_i = df_X.iloc[:lo, list_columns_always_use_t_location]
            Y_train_i = df_Y.iloc[:lo, key_list[position]]
            X_test_i = df_X.iloc[lo_test:, list_columns_always_use_t_location]
            ini_model=timer()
            
            lista_errores_semana=[]
            
            reg = XGBRegressor(**best_parameters[position]).fit(X_train_i, Y_train_i)
            lista_errores_semana.append(((Y_train_i - reg.predict(X_train_i)) ** 2).mean())
            MAPE_train=MAPE(Y_train_i,reg.predict(X_train_i))
            MAPE_train5000=MAPE(Y_train_i,reg.predict(X_train_i),5000)
            Yhat_test_all_XGB[lo - index_year_week_train - 1:hi - index_year_week_train - 1, 0] = reg.predict(
                            X_test_i).reshape((-1))
            
            fin_model=timer()-ini_model
            
            sys.stdout.write(f'Tunning parameters:{best_parameters[position]}')
            print(' ')
            sys.stdout.write(f'MSE train:{lista_errores_semana[-1]:02.3f}, '
                             +f', MAPE train:{MAPE_train:02.3f}%, '
                             +f',sin threshold MAPE train:{MAPE_train5000:02.3f}%, '
                             +f', Time train:{fin_model:02.3f} s.')
            
            
            print(' ')    
            print("READY WEEK!")
            print(' ')
            
            year_test=df_final.año.max()
            week_test=df_final[df_final.año == year_test].semana_año.max()
            reg.save_model('save_model/'+str(year_test)+'_'+str(week_test)+'_t_'+str(t)+'_target_'+str(target)+'.model')
            reg.save_model('save_model/last_model/last_model_t_'+str(t)+'_target_'+str(target)+'.model')
            Y_test=pd.DataFrame(Y_test) #suppose is null
            
            Y_test.loc[:, "XGB_t_" + str(t)+'_target_'+ str(target)] = Yhat_test_all_XGB[:, 0]
    
            lista_aux=['codigo','start_date', 'año', 'mes', 'semana_año', 'semana_mes', 'dia_año', 'dia_mes', 'dia_semana', 'bloque']
            for i in range(len(lista_aux)):
                Y_test[lista_aux[i]] = list_programa[i][index_year_week_train + 1:]
    
            Y_test.to_csv("output\pred_for_last_week_data\week_"+str(t)+"/model_predict_xgb_tunning_"+str(year_test)+"_"+str(week_test)+"_target_"+str(target)+".csv",index=False)
            
            
            plot_importance_columns_last_week(reg,X_test_i,year_test,week_test,t,target)
            
            sys.stdout.write(f'Total time for target {target} and Time step {t} (m):{fin_model/60:02.3f} .')
            print(' ')
            print('##----------------------####-------------------------##')
    
    fin= (timer()-inicio)/60
    print('Predict ready! time(m): ',round(fin,2))
    
    
   
def pred_X_test(df):
    inicio=timer()
    lista_columns = []
    with open("input\columns_predict.txt", "r") as f:
        for line in f:
            lista_columns.append(line.strip())
    df_final,list_programa=generate_data(tunning=False,save_metadata=False,other_df=df,for_pred=True)
    fin= timer()-inicio
    print('Data ready! time(s): ',round(fin,2))
    lista_columns=[s+'_5' for s in lista_columns]
    df_X = df_final[df_final.columns.difference(lista_columns)].copy()
    row, col = df_X.shape
    dic_aux={0:(1,0),1:(1,1),2:(2,0),3:(2,1),4:(3,0),5:(3,1),6:(4,0),7:(4,1),8:(5,0),9:(5,1),10:(6,0),11:(6,1)}
            
    print('Time to train model!')
    inicio=timer()

    key_list = list(dic_aux.keys())
    val_list = list(dic_aux.values())
    
    #call columns
    dict_colums=json.load(open('input/columns_train_models/columns_train_models.json', 'r'))
    dict_colums= {int(k):v for k,v in dict_colums.items()}
    """
    list_columns_always_use=[col for col in df_X.columns if '_t_' not in col]
    list_columns_always_use_location=[df_X.columns.get_loc(c) for c in list_columns_always_use ]
    """
    
    Yhat_test = np.zeros((row,len(dic_aux)))
    for t in range(1, 7):
        print("Time step predict: ", t )
        print('**--------------------------------------------**')
        print(' ')
        # Run sliding window regression models
        for target in range(2):
            print("Target :", target )
            position = val_list.index((t,target))
            list_columns_always_use_t_location=[df_X.columns.get_loc(c) for c in dict_colums[position]]
            """
            list_columns_use_t=[col for col in df_X.columns if '_t_'+str(t)+'_' in col]
            list_columns_always_use_t_location=[df_X.columns.get_loc(c) for c in list_columns_use_t ]
            list_columns_always_use_t_location.extend(list_columns_always_use_location)
            """
            X_test= df_X.iloc[:, list_columns_always_use_t_location]
            reg = XGBRegressor()
            reg.load_model('save_model/last_model/last_model_t_'+str(t)+'_target_'+str(target)+'.model')
            Yhat_test[:, position]=reg.predict(X_test)#reshape((-1))
    
    Y_test = pd.DataFrame(Yhat_test)
    Y_test.columns=['model_t_'+str(t)+'_target_'+str(target) for t,target in val_list]
    lista_aux=['codigo','start_date', 'año', 'mes', 'semana_año', 'semana_mes', 'dia_año', 'dia_mes', 'dia_semana', 'bloque']
    for i in range(len(lista_aux)):
        Y_test[lista_aux[i]] = list_programa[i]
    
    
    return Y_test