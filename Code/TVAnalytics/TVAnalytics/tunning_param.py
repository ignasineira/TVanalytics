# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 19:14:28 2021

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

#from statsmodels.tools.eval_measures import rmse

import itertools



from .data_processing import generate_data,generate_data_time_slot



## pipeline_tunning_param(param_tuning,save_metadata=True, n_week_train=15,start_year=2015,start_month=1,generate_new_df=True,week_list=[1,2,3,4,5,6], target_list=[0])
## pipeline_tunning_param(param_tuning,save_metadata=False, n_week_train=15,start_year=2015,start_month=1,generate_new_df=False,week_list=[1,2,3,4,5,6], target_list=[1])



def pipeline_tunning_param(param_tuning,save_metadata=False, n_week_train=15,start_year=2015,start_month=1,generate_new_df=False,week_list=[1,2,3,4,5,6], target_list=[0,1]):
    """
    genereate data, filter data from the last value is not na for t+6 target 
    
    """
    #list of columns predict
    inicio=timer()
    lista_columns = []
    with open("input\columns_predict.txt", "r") as f:
        for line in f:
            lista_columns.append(line.strip())
    
    if generate_new_df:
        df_final=generate_data(start_year,start_month,save_metadata=save_metadata,tunning=True)
        df_final.to_csv('input/df_final.csv',index = False)
    else:
        df_final=pd.read_csv('input/df_final.csv')
    fin= timer()-inicio
    print('Data ready! time(s): ',round(fin,2))
    lista_columns=[s+'_5' for s in lista_columns]
    
    #filter data from the last value is not na for t+6 target
    last_value = df_final[~df_final[lista_columns[-2]].isna()][['semana_año','año']].index[-1]
    #filter data
    df_final = df_final.loc[:last_value,:].copy()
    
    df_X = df_final[df_final.columns.difference(lista_columns)].copy()
    df_Y = df_final[lista_columns[::2]].copy()

    #split data
    list_tuple_week = list(set((i, j) for i, j in zip(df_final.año, df_final.semana_año)))
    list_tuple_week.sort()

    list_index = []
    for tuplas in list_tuple_week:
        list_index.append(df_final[(df_final.año == tuplas[0]) & (df_final.semana_año == tuplas[1])].index[-1])
        
    
    year_train,week_train=list_tuple_week[-(n_week_train+1)]
    index_year_week_train = df_final[(df_final.año == year_train) & (df_final.semana_año == week_train)].index[-1]


   
    
    keys, values = zip(*param_tuning.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    #save the all the combinations in a list
    with open('save_model/tunning_parameters/param_tuning.txt', 'w') as f:
        for item in permutations_dicts:
            f.write("%s\n" % item)
    num_model=len(permutations_dicts)
    print('Time to train model!')
    inicio=timer()
    train_model(df_X,df_Y,num_model,permutations_dicts,list_tuple_week,index_year_week_train,list_index,preprocessing=0, week_list=week_list, target_list=target_list)
    fin= (timer()-inicio)/60
    print('Predict ready! time(m): ',round(fin,2))


def pipeline_tunning_param_time_slot(param_tuning,save_metadata=False, n_week_train=15,start_year=2015,start_month=1,generate_new_df=False,week_list=[1,2,3,4,5,6], target_list=[0,1]):
    """
    genereate data, filter data from the last value is not na for t+6 target 
    
    
    param_tuning: is a dict of dict ( for each time slot one param_tuning )
    
    """
    #list of columns predict
    inicio=timer()
    lista_columns = []
    with open("input\columns_predict.txt", "r") as f:
        for line in f:
            lista_columns.append(line.strip())
    
    
    if generate_new_df:
         dict_df_final=generate_data_time_slot(start_year,start_month,save_metadata=save_metadata,tunning=True)
         for key, df in dict_df_final.items():
             df.to_csv('input/time_slot/df_final_'+key+'.csv',index = False)
    else:
        dict_df_final={}
        arr_folder = os.listdir('input/time_slot/')
        arr = [ item for item in arr_folder ] 
        for item in arr:
            df_aux= pd.read_csv('input/time_slot/'+item,sep=',')
            franja=item[len("df_final_"):]
            dict_df_final[franja]=df_aux
    fin= timer()-inicio
    print('Data ready! time(s): ',round(fin,2))
    lista_columns=[s+'_5' for s in lista_columns]
    inicio_all=timer()
    for key, df_final in dict_df_final.items():
        #filter data from the last value is not na for t+6 target
        last_value = df_final[~df_final[lista_columns[-2]].isna()][['semana_año','año']].index[-1]
        #filter data
        df_final = df_final.loc[:last_value,:].copy()
        
        df_X = df_final[df_final.columns.difference(lista_columns)].copy()
        df_Y = df_final[lista_columns[::2]].copy()
    
        #split data
        list_tuple_week = list(set((i, j) for i, j in zip(df_final.año, df_final.semana_año)))
        list_tuple_week.sort()
    
        list_index = []
        for tuplas in list_tuple_week:
            list_index.append(df_final[(df_final.año == tuplas[0]) & (df_final.semana_año == tuplas[1])].index[-1])
            
        
        year_train,week_train=list_tuple_week[-(n_week_train+1)]
        index_year_week_train = df_final[(df_final.año == year_train) & (df_final.semana_año == week_train)].index[-1]
        
        keys, values = zip(*param_tuning[key].items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        #save the all the combinations in a list
        with open('save_model/tunning_parameters/param_tuning_'+key+'.txt', 'w') as f:
            for item in permutations_dicts:
                f.write("%s\n" % item)
        num_model=len(permutations_dicts)
        print('Time to train model!')
        print('Time slot: '+key)
        inicio=timer()
        train_model(df_X,df_Y,num_model,permutations_dicts,list_tuple_week,index_year_week_train,list_index,preprocessing=0, week_list=week_list, target_list=target_list, time_slot=True, file_name=key)
        fin= (timer()-inicio)/60
        print('Predict ready! time(m): ',round(fin,2))
    fin_all= (timer()-inicio_all)/60
    print('Final Predict ready! time(m): ',round(fin_all,2))

def train_model(df_X,df_Y,num_model,permutations_dicts,list_tuple_week,index_year_week_train,list_index,preprocessing=0, week_list=[1,2,3,4,5,6], target_list=[0,1], time_slot=False, file_name=None):
    """
    week_list: choose which week you want to train (1,2,3,4,5,6)
    target_list : choose which target  you want to train  ( 0 or 1)
    """
    
    
    
    preprocessing=str(preprocessing)
    dic_aux={0:(1,0),1:(1,1),2:(2,0),3:(2,1),4:(3,0),5:(3,1),6:(4,0),7:(4,1),8:(5,0),9:(5,1),10:(6,0),11:(6,1)}
    key_list = list(dic_aux.keys())
    val_list = list(dic_aux.values())
    
    list_columns_always_use=[col for col in df_X.columns if '_t_' not in col]
    list_columns_always_use_location=[df_X.columns.get_loc(c) for c in list_columns_always_use ]
    for t in week_list:
        print("Time step predict: ", t )
        print('**--------------------------------------------**')
        print(' ')
        # Run sliding window regression models
        for target in target_list:
            
            print("Target :", target )
            print("Number of model to predict:",num_model)
            position = val_list.index((t,target))
            # Define test data
            #X_test = df_X.iloc[ index_year_week_train+ 1:, :]
            #Y_test = df_Y.iloc[ index_year_week_train + 1:, key_list[position]]
            #Yhat_test = np.zeros_like(Y_test)
            #Yhat_test_all_XGB = np.zeros((Yhat_test.shape[0], num_model))
            #list_errores = []
            tiempo_training=0
            aux_num_model=0
    
            # Iterate list of index
            for i in range(list_index.index(index_year_week_train), len(list_index) - 1):
                print("Training for test point:", list_tuple_week[i])
                print("Test point:", list_tuple_week[i + 1])
                # Set training window
                list_columns_use_t=[col for col in df_X.columns if '_t_'+str(t)+'_' in col]
                list_columns_always_use_t_location=[df_X.columns.get_loc(c) for c in list_columns_use_t ]
                list_columns_always_use_t_location.extend(list_columns_always_use_location)
                lo = list_index[i] + 1
                hi = list_index[i + 1] + 1
                X_train_i = df_X.iloc[:lo, list_columns_always_use_t_location]
                Y_train_i = df_Y.iloc[:lo, key_list[position]]
                X_test_i = df_X.iloc[lo:hi, list_columns_always_use_t_location]
                Y_test_i = df_Y.iloc[lo:hi, key_list[position]]
                Yhat_test_all_XGB_i = np.zeros((Y_test_i.shape[0], num_model))
                #print(X_train_i.shape,Y_train_i.shape,X_test_i.shape,Y_test_i.shape)
                lista_errores_semana = []
    
                for j in range(num_model):
                    ini_model=timer()
                    aux_num_model+=1
                    reg = XGBRegressor(**permutations_dicts[j]).fit(X_train_i, Y_train_i)# early_stopping_rounds=5
                    lista_errores_semana.append(((Y_train_i - reg.predict(X_train_i)) ** 2).mean())
                    lista_errores_semana.append(((Y_test_i - reg.predict(X_test_i)) ** 2).mean())
                    
                    MAPE_train=MAPE(Y_train_i,reg.predict(X_train_i))
                    MAPE_test=MAPE(Y_test_i,reg.predict(X_test_i))
                    MAPE_train5000=MAPE(Y_train_i,reg.predict(X_train_i),5000)
                    MAPE_test5000=MAPE(Y_test_i,reg.predict(X_test_i),5000)
                    Yhat_test_all_XGB_i[:, j] = reg.predict(X_test_i).reshape((-1))
                    
                    fin_model=timer()-ini_model
                    lista_errores_semana.append(fin_model)
                    tiempo_training+=fin_model
                    
                    sys.stdout.write(f'Tunning parameters:{permutations_dicts[j]}')
                    print(' ')
                    sys.stdout.write(f'MSE train:{lista_errores_semana[-3]:02.3f}, '
                                     +f', MSE test:{lista_errores_semana[-2]:02.3f}, '
                                     +f', MAPE train:{MAPE_train:02.3f}%, '
                                     +f', MAPE test:{MAPE_test:02.3f}%, '
                                     +f',sin threshold MAPE train:{MAPE_train5000:02.3f}%, '
                                     +f', MAPE test:{MAPE_test5000:02.3f}%, '
                                     +f', Time train:{fin_model:02.3f} s., '
                       + f'Avg-Time:{tiempo_training/(aux_num_model):02.3f} s.\n')
                    
                Y_test_i = pd.DataFrame(Y_test_i)
                for k in range(num_model):
                   Y_test_i.loc[:, "XGB_" + str(k)] = Yhat_test_all_XGB_i[:, k]
                   
                if time_slot:
                    Y_test_i.to_csv("output\All_tunning_parameter_time_slot\week_"+str(t)+"\model_predict_xgb_tunning_target_"+str(target)+"_pre_"+preprocessing+"_year_"+str(list_tuple_week[i + 1][0])+"_week_"+str(list_tuple_week[i + 1][1])+"_franja_"+file_name+".csv")

                else:
                 Y_test_i.to_csv("output\All_tunning_parameter\week_"+str(t)+"\model_predict_xgb_tunning_target_"+str(target)+"_pre_"+preprocessing+"_year_"+str(list_tuple_week[i + 1][0])+"_week_"+str(list_tuple_week[i + 1][1])+".csv")
                
                print("READY WEEK!")
                print(lista_errores_semana[2::3])
                print('='*60) 
                
            sys.stdout.write(f'Total time for target {target} and Time step {t} (m):{tiempo_training/60:02.3f} .')
            print(' ')
            print('##----------------------####-------------------------##')



def MAPE(Y_i,Y_hat,threshold=500):
    MAPE=(Y_i -Y_hat)/Y_i
    MAPE[~np.isfinite(MAPE)]=np.nan
    MAPE=MAPE[~np.isnan(MAPE)]
    MAPE=np.where((100*MAPE).abs()>threshold , threshold,(100*MAPE).abs()).mean().round(4)
    return MAPE

def normalize(data, train_split):
    data_min = data[:train_split].min(axis=0)
    data_max = data[:train_split].max(axis=0)
    return (data - data_min) / (data_max-data_min)




