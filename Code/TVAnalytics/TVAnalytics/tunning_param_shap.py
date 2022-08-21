# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 14:21:11 2021

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

from .data_processing import generate_data



from xgboost import plot_importance, plot_tree
import shap
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from matplotlib import pyplot
# load JS visualization code to notebook
shap.initjs()

"""
shap 

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
explainer = shap.TreeExplainer(reg)

***Most importan feature for xgboost

1. Call df_X for tunning
2. Call xgb model
3. Call columns for each tuple (target,week)
4. Call best tunning parameters

"""



def pipeline_tunning_param_shap(lower_bound=0.001, n_week_train=15):
    
    """
    This stage assumes that parameter tuning was performed and 
    the best set of hyperparameters was chosen. 
    
    input:
        
    import: 
        1. Call df_X for tunning
        2. Call xgb model
        3. Call columns for each tuple (target,week)**
        4. Call best tunning parameters
        
    output:
    
    
    
    """
    
    path = 'C:/Users/ignasi/Desktop/TVAnalytics'
    
    os.chdir(path)
    inicio=timer()
    df_final=pd.read_csv('input/df_final.csv')
    lista_columns = []
    with open("input\columns_predict.txt", "r") as f:
        for line in f:
            lista_columns.append(line.strip())
    
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
    
    #load best parameters
    best_parameters=[]
    dic_aux={}
    for t in range(1,7):
        for i in range(2):
            best_parameters.append(json.load(open('save_model/tunning_parameters/best_parameter_t_'+str(t)+'_target_'+str(i)+'.json', 'r')))
            dic_aux[len(best_parameters)-1]=(t,i)
            
    key_list = list(dic_aux.keys())
    val_list = list(dic_aux.values())
    
    list_columns_always_use=[col for col in df_X.columns if '_t_' not in col]
    list_columns_always_use_location=[df_X.columns.get_loc(c) for c in list_columns_always_use ]
    
    dict_colums_shap = {}
    
    for t in range(1, 7):
        print("Time step predict: ", t )
        print('**--------------------------------------------**')
        print(' ')
        # Run sliding window regression models
        for target in range(2):    
            print("Target :", target )
            position = val_list.index((t,target))
            list_columns_use_t=[col for col in df_X.columns if '_t_'+str(t)+'_' in col]
            list_columns_always_use_t_location=[df_X.columns.get_loc(c) for c in list_columns_use_t ]
            list_columns_always_use_t_location.extend(list_columns_always_use_location)
            lo = list_index[-2] + 1
            X_train = df_X.iloc[:lo, list_columns_always_use_t_location]
            Y_train = df_Y.iloc[:lo, key_list[position]]
            
            #train the model
            reg = XGBRegressor(**best_parameters[position]).fit(X_train, Y_train)
            # explain the model's predictions using SHAP
            explainer = shap.TreeExplainer(reg)
            shap_values = explainer.shap_values(X_train)
            
            feature_names =[df_X.columns[c] for c in list_columns_always_use_t_location ] 
            max_display= len(X_train.columns)
            
            feature_order = np.argsort(np.sum(np.abs(shap_values),axis=0))
            feature_order = feature_order[-min(max_display,len(feature_order)):]
            
            feature_inds = feature_order[:max_display]
            #y_pos = np.arange(len(feature_inds))
            
            df_shap_values =np.abs(shap_values).mean(0)
            
            shap_importance = pd.DataFrame(df_shap_values[feature_inds], [feature_names[i] for i in feature_inds],
                                           columns=['Importance'])
            #filter columns
            
            shap.summary_plot(shap_values,
                  features = X_train,
                  feature_names=feature_names,
                 plot_type='bar',
                 color='dodgerblue',show=False)

            plt.savefig("image\Shap_tunning\week_"+str(t)+"\plot_summary_importance_columns_shap_target_"+str(target)+"_t_"+str(t)+"_summary_plot2.png",dpi=150, bbox_inches='tight')
            
            columns_shap=shap_importance[shap_importance['Importance']>0.0]
            columns_shap.to_csv("output\Shap_tunning\week_"+str(t)+"\summary_importance_columns_shap_target_"+str(target)+"_t_"+str(t)+"_lower_bound_"+str(0.0)+".csv")
            columns_shap=shap_importance[shap_importance['Importance']>lower_bound]
            columns_shap.to_csv("output\Shap_tunning\week_"+str(t)+"\summary_importance_columns_shap_target_"+str(target)+"_t_"+str(t)+"_lower_bound_"+str(lower_bound)+".csv")
            
            
            #save columns in a dict
            dict_colums_shap[position]=list(columns_shap.index.values) 
            
            print("The first model is trained with number of columns: "+ str(max_display))
            print("The shap model is trained with number of columns: "+ str(len(columns_shap)))
    
    #save the columns
    with open('output/Shap_tunning/columns_shap.json', 'w') as fp:
        json.dump(dict_colums_shap, fp)
        
    
    for t in range(1,7):
        print("Time step predict: ", t )
        print('**--------------------------------------------**')
        print(' ')
        # Run sliding window regression models
        for target in range(2):
            
            print("Target :", target )
            position = val_list.index((t,target))
    
            # Iterate list of index
            for i in range(list_index.index(index_year_week_train), len(list_index) - 1):
                print("Training for test point:", list_tuple_week[i])
                print("Test point:", list_tuple_week[i + 1])
                # Set training window
                list_columns_use_t=dict_colums_shap[position]
                list_columns_always_use_t_location=[df_X.columns.get_loc(c) for c in list_columns_use_t ]
                lo = list_index[i] + 1
                hi = list_index[i + 1] + 1
                X_train_i = df_X.iloc[:lo, list_columns_always_use_t_location]
                Y_train_i = df_Y.iloc[:lo, key_list[position]]
                X_test_i = df_X.iloc[lo:hi, list_columns_always_use_t_location]
                Y_test_i = df_Y.iloc[lo:hi, key_list[position]]
                
                Yhat_test_i = np.zeros(Y_test_i.shape)
                ini_model=timer()
                lista_errores_semana=[]
                reg = XGBRegressor(**best_parameters[position]).fit(X_train_i, Y_train_i)
                lista_errores_semana.append(((Y_train_i - reg.predict(X_train_i)) ** 2).mean())
                MAPE_train=MAPE(Y_train_i,reg.predict(X_train_i))
                MAPE_test=MAPE(Y_test_i,reg.predict(X_test_i))
                MAPE_train5000=MAPE(Y_train_i,reg.predict(X_train_i),5000)
                MAPE_test5000=MAPE(Y_test_i,reg.predict(X_test_i),5000)
                Yhat_test_i[:] = reg.predict(
                                X_test_i).reshape((-1))
                
                fin_model=timer()-ini_model
                
                sys.stdout.write(f'Tunning parameters:{best_parameters[position]}')
                print(' ')
                sys.stdout.write(f'MSE train:{lista_errores_semana[-1]:02.3f}, '
                                 +f', MAPE train:{MAPE_train:02.3f}%, '
                                 +f', MAPE test:{MAPE_test:02.3f}%,'
                                 +f',sin threshold MAPE train:{MAPE_train5000:02.3f}%, '
                                 +f', MAPE test:{MAPE_test5000:02.3f}%, '
                                 +f', Time train:{fin_model:02.3f} s.')
            
            
                Y_test_i = pd.DataFrame(Y_test_i)
                Y_test_i.loc[:, "XGB_0"] = Yhat_test_i[:]
                Y_test_i.to_csv("output\Shap_tunning\week_"+str(t)+"\model_predict_xgb_tunning_target_"+str(target)+"_pre_0_year_"+str(list_tuple_week[i + 1][0])+"_week_"+str(list_tuple_week[i + 1][1])+".csv")
                
                print("READY WEEK!")
                print('='*60) 
                print(' ')
                print('##----------------------####-------------------------##')
    
    fin= (timer()-inicio)/60
    print('Predict ready! time(m): ',round(fin,2))
    

    
def MAPE(Y_i,Y_hat,threshold=500):
    MAPE=(Y_i -Y_hat)/Y_i
    MAPE[~np.isfinite(MAPE)]=np.nan
    MAPE=MAPE[~np.isnan(MAPE)]
    MAPE=np.where((100*MAPE).abs()>threshold , threshold,(100*MAPE).abs()).mean().round(4)
    return MAPE   
            
    
            
        
            
    





