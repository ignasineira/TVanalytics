# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 16:13:51 2021

@author: ignasi
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import plotly.express as px

#from plotly.subplots import make_subplots
#import plotly.graph_objects as go

#import plotly.offline as py
#py.plot(fig, filename='image/graph.png')
pd.set_option("display.precision", 8)

import os 
import ast
import json

def main_select_best_params(time_slot=False):
    
    if time_slot:
        week_list = list(range(1, 2))
        target_list = [0,1]
        path='output/All_tunning_parameter_time_slot/week_'
        franja = pd.read_csv('input/HORA_FRANJA_ID_map.csv', sep=';')
        list_franjas=franja['time_slot'].unique().tolist()
        for week in week_list:
            for target in target_list:
                arr_folder = os.listdir(path+str(week)+'/')
                arr = [ item for item in arr_folder if 'target_'+str(target)+'_' in item ]
                for group_name in list_franjas:
    
                    arr_franja = [ item for item in arr if group_name in item ]
                    df=[]
                
                    for item in arr_franja:
                        df_aux= pd.read_csv(path+str(week)+'/'+item,sep=',')
                        df.append(df_aux)
                
                
                    df=pd.concat(df)
            
                    df.drop(columns=['Unnamed: 0'], inplace= True)
                
                    list_programa=pd.read_csv('output/All_tunning_parameter_time_slot/X_data_test_'+group_name+'.csv')
                
                    list_programa=list_programa.iloc[list_programa.index[-df.shape[0]:],:]
                    df[list_programa.columns.tolist()]= list(list_programa.values)
                
                    df.start_date = pd.to_datetime(df.start_date)
                    df.sort_values(by=['dia_año','bloque'], inplace=True)
                    df.drop(columns=['año', 'mes', 'semana_mes', 'dia_año', 'dia_mes','holiday',
                                     'holiday_weekend', 'days_next_holiday', 
                                     'days_next_holiday_weekend'],inplace=True)
                
                    if target == 0:
                        df.rename(columns={"indm25_64_t_"+str(week)+"_5": "indm25_64"},inplace=True)
                        column="indm25_64"
                    else:
                        df.rename(columns={"indhmabc25_64_t_"+str(week)+"_5": "indhmabc25_64"},inplace=True)
                        column="indhmabc25_64"
        
                    bloque_hora=pd.read_csv('input/BLOQUES_ID_map.csv',sep=';')
                    bloque_hora['HORA']=pd.to_datetime(bloque_hora['HORA']).dt.time
                    bloque_hora['HORA']=bloque_hora['HORA'].apply(lambda t: t.strftime('%H:%M'))
                    bloque_hora.rename(columns={'BLOQUES':'bloque'},inplace=True)
                    df=df.merge(bloque_hora, on=['bloque'])
                    df=df[df[column]!=0.0]
                    
                    error_v2_time_slot(df,column,target,week,group_name, shap_tunning=False)
        
    
    else:
    
        week_list = list(range(1, 7))
        target_list = [0,1]
        
        path='output/All_tunning_parameter/week_'
        
        for week in week_list:
            for target in target_list:
        
                arr_folder = os.listdir(path+str(week)+'/')
                arr = [ item for item in arr_folder if 'target_'+str(target)+'_' in item ]
        
                df=[]
            
                for item in arr:
                    df_aux= pd.read_csv(path+str(week)+'/'+item,sep=',')
                    df.append(df_aux)
                
                
                df=pd.concat(df)
            
                df.drop(columns=['Unnamed: 0'], inplace= True)
                
                list_programa=pd.read_csv('output/All_tunning_parameter/X_data_test.csv')
                
                list_programa=list_programa.iloc[list_programa.index[-df.shape[0]:],:]
                df[list_programa.columns.tolist()]= list(list_programa.values)
                
                df.start_date = pd.to_datetime(df.start_date)
                df.sort_values(by=['dia_año','bloque'], inplace=True)
                df.drop(columns=['año', 'mes', 'semana_mes', 'dia_año', 'dia_mes','holiday',
                                 'holiday_weekend', 'days_next_holiday', 
                                 'days_next_holiday_weekend'],inplace=True)
                
                if target == 0:
                    df.rename(columns={"indm25_64_t_"+str(week)+"_5": "indm25_64"},inplace=True)
                    column="indm25_64"
                else:
                    df.rename(columns={"indhmabc25_64_t_"+str(week)+"_5": "indhmabc25_64"},inplace=True)
                    column="indhmabc25_64"
        
                bloque_hora=pd.read_csv('input/BLOQUES_ID_map.csv',sep=';')
                bloque_hora['HORA']=pd.to_datetime(bloque_hora['HORA']).dt.time
                bloque_hora['HORA']=bloque_hora['HORA'].apply(lambda t: t.strftime('%H:%M'))
                bloque_hora.rename(columns={'BLOQUES':'bloque'},inplace=True)
                df=df.merge(bloque_hora, on=['bloque'])
                df=df[df[column]!=0.0]
                
                error_v2(df,column,target,week, shap_tunning=False)

def main_select_best_params_after_shap():
        
    week_list = list(range(1, 7))
    target_list = [0,1]
    
    path='output/Shap_tunning/week_'
    
    for week in week_list:
        for target in target_list:
    
            arr_folder = os.listdir(path+str(week)+'/')
            arr = [ item for item in arr_folder if 'target_'+str(target)+'_' in item ]
    
            df=[]
        
            for item in arr:
                df_aux= pd.read_csv(path+str(week)+'/'+item,sep=',')
                df.append(df_aux)
            
            
            df=pd.concat(df)
        
            df.drop(columns=['Unnamed: 0'], inplace= True)
            
            list_programa=pd.read_csv('output/All_tunning_parameter/X_data_test.csv')
            
            list_programa=list_programa.iloc[list_programa.index[-df.shape[0]:],:]
            df[list_programa.columns.tolist()]= list(list_programa.values)
            
            df.start_date = pd.to_datetime(df.start_date)
            df.sort_values(by=['dia_año','bloque'], inplace=True)
            df.drop(columns=['año', 'mes', 'semana_mes', 'dia_año', 'dia_mes','holiday',
                             'holiday_weekend', 'days_next_holiday', 
                             'days_next_holiday_weekend'],inplace=True)
            
            if target == 0:
                df.rename(columns={"indm25_64_t_"+str(week)+"_5": "indm25_64"},inplace=True)
                column="indm25_64"
            else:
                df.rename(columns={"indhmabc25_64_t_"+str(week)+"_5": "indhmabc25_64"},inplace=True)
                column="indhmabc25_64"
    
            bloque_hora=pd.read_csv('input/BLOQUES_ID_map.csv',sep=';')
            bloque_hora['HORA']=pd.to_datetime(bloque_hora['HORA']).dt.time
            bloque_hora['HORA']=bloque_hora['HORA'].apply(lambda t: t.strftime('%H:%M'))
            bloque_hora.rename(columns={'BLOQUES':'bloque'},inplace=True)
            df=df.merge(bloque_hora, on=['bloque'])
            df=df[df[column]!=0.0]
            
            error_v2(df,column,target,week, shap_tunning=True)


def error_metrics(x,XGB_models,column):
    d = {}
    #d['PROGRAMA']=df.PROGRAMA
    
    #MSE_global,MAE_GLOBAL, MAPE_GLOBAL,WMAPE_GLOBAL =0,0,0,0
    for i in range(XGB_models):
        d['MSE_'+str(i)] = ((x[column] - x['XGB_'+str(i)]) ** 2).mean()
        d['MAE_'+str(i)] = ((x[column] - x['XGB_'+str(i)]).abs()).mean()
        d['MAPE_'+str(i)]=(round(100*(x[column]-x['XGB_'+str(i)])/x[column],2).abs()).mean()
        # make a series called mape
        se_mape = (round(100*(x[column]-x['XGB_'+str(i)])/x[column],2).abs())
        # get a float of the sum of the actual
        ft_actual_sum = x[column].sum()
        # get a series of the multiple of the actual & the mape
        se_actual_prod_mape = x[column] * se_mape
        # summate the prod of the actual and the mape
        ft_actual_prod_mape_sum = se_actual_prod_mape.sum()
        # float: wmape of forecast
        d['WMAPE_'+str(i)]= ft_actual_prod_mape_sum / ft_actual_sum

    
    return pd.Series(d, index=d.keys())



def error_v2(df,column,target,week, shap_tunning=False):
    path='output/All_tunning_parameter/main_result/'
    XGB_models = sum('XGB_' in s for s in df.columns) 
    
    error=df.groupby(by=['semana_año']).apply(lambda x:error_metrics(x,XGB_models,column)).reset_index(0)
    

    error=error.groupby(by=['semana_año']).agg('mean').round(2).reset_index()
    aux=error[sorted(error.columns.tolist())].describe().transpose()
    aux = aux[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]

    if shap_tunning:
        dict_colums_train_models=json.load(open('input/columns_train_models/columns_train_models.json', 'r'))
        dict_colums_train_models = {int(k):v for k,v in dict_colums_train_models.items()}
        
        dict_colums_shap=json.load(open('output/Shap_tunning/columns_shap.json', 'r'))
        dict_colums_shap = {int(k):v for k,v in dict_colums_shap.items()}
        
        dic_aux={0:(1,0),1:(1,1),2:(2,0),3:(2,1),4:(3,0),5:(3,1),6:(4,0),7:(4,1),8:(5,0),9:(5,1),10:(6,0),11:(6,1)}
        key_list = list(dic_aux.keys())
        val_list = list(dic_aux.values())
        position = val_list.index((week,target))
        
        value = aux[aux.index.isin(['WMAPE_'+str(i) for i in range(XGB_models)])].sort_values(by=['mean'])['mean'][0]
        
        #call tunning value
        with open (path+'mejor_resultados_tunning_target_'+str(target)+'_t_'+str(week)+'.txt','r') as f:
            value_tunning =f.readline() #aux[aux.index.isin(['WMAPE_'+str(i) for i in range(XGB_models)])].sort_values(by=['mean'])['mean'][0]
            value_tunning =float(value_tunning)
        
        if value <= value_tunning:
            dict_colums_train_models[position]=dict_colums_shap[position]
            print("Keep the shap option , number of columns:"+ str(len(dict_colums_shap[position])))
            #save the columns
            with open('input/columns_train_models/columns_train_models.json', 'w') as fp:
                json.dump(dict_colums_train_models, fp)
            
        else:
            
            train_columns=pd.read_csv('input\columns_name_post_pipeline.csv', sep=',').columns.tolist()
            lista_columns = []
            with open("input\columns_predict.txt", "r") as f:
                for line in f:
                    lista_columns.append(line.strip())
            lista_columns=[s+'_5' for s in lista_columns]       
            train_columns=[col for col in train_columns if col not in lista_columns]
            list_columns_always_use=[col for col in train_columns if '_t_' not in col]
            list_columns_use_t=[col for col in train_columns if '_t_'+str(week)+'_' in col and col not in lista_columns ]
            list_columns_use_t.extend(list_columns_always_use)
            print("Keep the firt option , number of columns:"+ str(len(list_columns_use_t)))
            dict_colums_train_models[position]=list_columns_use_t
            #save the columns
            with open('input/columns_train_models/columns_train_models.json', 'w') as fp:
                json.dump(dict_colums_train_models, fp)
            
    
    else:
        file = open('save_model/tunning_parameters/param_tuning.txt', 'r') 
        list_dict = []
        for x in file:
            list_dict.append(ast.literal_eval(x.replace('\n','')))
        
        aux= aux[aux.index.isin(['WMAPE_'+str(i) for i in range(XGB_models)])].copy()
        df_dict = pd.DataFrame.from_dict(list_dict, orient='columns')
        aux[df_dict.columns.tolist()] = df_dict.values
        
        aux = aux [['learning_rate',
               'max_depth', 'n_estimators', 'n_jobs', 'colsample_bytree',
               'colsample_bylevel','mean', 'std', 'min', '25%', '50%', '75%', 'max']].copy()
        
        print("For target: " +column)
        print ("Week :" + str(week))
         #save the best values
        best_values = list_dict[int(aux[aux.index.isin(['WMAPE_'+str(i) for i in range(XGB_models)])].sort_values(by=['mean']).index.tolist()[0].split("_")[-1])]
        print(best_values)
        with open('save_model/tunning_parameters/best_parameter_t_'+str(week)+'_target_'+str(target)+'.json', 'w') as fp:
            json.dump(best_values,fp)
        #Save values 
        aux.to_csv(path+'resultados_tunning_target_'+str(target)+'_t_'+str(week)+'.csv', index=False, sep=';',decimal=',')
        with open (path+'mejor_resultados_tunning_target_'+str(target)+'_t_'+str(week)+'.txt','w') as f:
            value = aux[aux.index.isin(['WMAPE_'+str(i) for i in range(XGB_models)])].sort_values(by=['mean'])['mean'][0]
            f.write(str(value))
    
def error_v2_time_slot(df,column,target,week,group_name, shap_tunning=False):
    path='output/All_tunning_parameter_time_slot/main_result/'
    XGB_models = sum('XGB_' in s for s in df.columns) 
    
    error=df.groupby(by=['semana_año']).apply(lambda x:error_metrics(x,XGB_models,column)).reset_index(0)
    

    error=error.groupby(by=['semana_año']).agg('mean').round(2).reset_index()
    aux=error[sorted(error.columns.tolist())].describe().transpose()
    aux = aux[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]

    if shap_tunning:
        dict_colums_train_models=json.load(open('input/columns_train_models/columns_train_models.json', 'r'))
        dict_colums_train_models = {int(k):v for k,v in dict_colums_train_models.items()}
        
        dict_colums_shap=json.load(open('output/Shap_tunning/columns_shap.json', 'r'))
        dict_colums_shap = {int(k):v for k,v in dict_colums_shap.items()}
        
        dic_aux={0:(1,0),1:(1,1),2:(2,0),3:(2,1),4:(3,0),5:(3,1),6:(4,0),7:(4,1),8:(5,0),9:(5,1),10:(6,0),11:(6,1)}
        key_list = list(dic_aux.keys())
        val_list = list(dic_aux.values())
        position = val_list.index((week,target))
        
        value = aux[aux.index.isin(['WMAPE_'+str(i) for i in range(XGB_models)])].sort_values(by=['mean'])['mean'][0]
        
        #call tunning value
        with open (path+'mejor_resultados_tunning_target_'+str(target)+'_t_'+str(week)+'.txt','r') as f:
            value_tunning =f.readline() #aux[aux.index.isin(['WMAPE_'+str(i) for i in range(XGB_models)])].sort_values(by=['mean'])['mean'][0]
            value_tunning =float(value_tunning)
        
        if value <= value_tunning:
            dict_colums_train_models[position]=dict_colums_shap[position]
            print("Keep the shap option , number of columns:"+ str(len(dict_colums_shap[position])))
            #save the columns
            with open('input/columns_train_models/columns_train_models.json', 'w') as fp:
                json.dump(dict_colums_train_models, fp)
            
        else:
            
            train_columns=pd.read_csv('input\columns_name_post_pipeline.csv', sep=',').columns.tolist()
            lista_columns = []
            with open("input\columns_predict.txt", "r") as f:
                for line in f:
                    lista_columns.append(line.strip())
            lista_columns=[s+'_5' for s in lista_columns]       
            train_columns=[col for col in train_columns if col not in lista_columns]
            list_columns_always_use=[col for col in train_columns if '_t_' not in col]
            list_columns_use_t=[col for col in train_columns if '_t_'+str(week)+'_' in col and col not in lista_columns ]
            list_columns_use_t.extend(list_columns_always_use)
            print("Keep the firt option , number of columns:"+ str(len(list_columns_use_t)))
            dict_colums_train_models[position]=list_columns_use_t
            #save the columns
            with open('input/columns_train_models/columns_train_models.json', 'w') as fp:
                json.dump(dict_colums_train_models, fp)
            
    
    else:
        file = open('save_model/tunning_parameters/param_tuning_'+group_name+'.txt', 'r') 
        list_dict = []
        for x in file:
            list_dict.append(ast.literal_eval(x.replace('\n','')))
        
        aux= aux[aux.index.isin(['WMAPE_'+str(i) for i in range(XGB_models)])].copy()
        df_dict = pd.DataFrame.from_dict(list_dict, orient='columns')
        aux[df_dict.columns.tolist()] = df_dict.values
        
        aux = aux [['learning_rate',
               'max_depth', 'n_estimators', 'n_jobs', 'colsample_bytree',
               'colsample_bylevel','mean', 'std', 'min', '25%', '50%', '75%', 'max']].copy()
        
        print("For target: " +column)
        print("For franja: " +group_name)
        print ("Week :" + str(week))
         #save the best values
        index=int(aux[aux.index.isin(['WMAPE_'+str(i) for i in range(XGB_models)])].sort_values(by=['mean']).index.tolist()[0].split("_")[-1])
        best_values = list_dict[index]
        print(best_values)
        with open('save_model/tunning_parameters/best_parameter_t_'+str(week)+'_target_'+str(target)+'_franja_'+group_name+'.json', 'w') as fp:
            json.dump(best_values,fp)
        #Save values 
        aux.to_csv(path+'resultados_tunning_target_'+str(target)+'_t_'+str(week)+'_franja_'+group_name+'.csv', index=False, sep=';',decimal=',')
        with open (path+'mejor_resultados_tunning_target_'+str(target)+'_t_'+str(week)+'_franja_'+group_name+'.txt','w') as f:
            value = aux[aux.index.isin(['WMAPE_'+str(i) for i in range(XGB_models)])].sort_values(by=['mean'])['mean'][0]
            f.write(str(value))
            
        list_XGB_models=['XGB_'+str(i) for i in range(XGB_models) if i!=index]
        keep_cloumns=[column for column in df.columns if column not in list_XGB_models]
        df_aux= df[keep_cloumns].copy()
        df_aux.rename(columns={"XGB_"+str(index): "XGB"},inplace=True)
        df_aux.to_csv(path+'best_model_'+str(target)+'_t_'+str(week)+'_franja_'+group_name+'.csv', index=False, sep=';',decimal=',')


"""
Falta guardar el valor de el obtenido , luego comparar si es mejor o peor que el shap
para eso tengo que cambiar las rutas de la funcion main_select_best_params
Crear un carpeta de columnas finales donde consulto las columnas de los modelos que voy a entrenar,
importante ver se debe cambiar las rutas de las columnas del data final. 

"""



def error(df,column,target,week):
    path='output/All_tunning_parameter/main_result/'
    
    error=df.copy()
    XGB_models = sum('XGB_' in s for s in df.columns) 
    for i in range(XGB_models):
        error['MAPE_'+str(i)]=round(100*(error[column]-error['XGB_'+str(i)])/error[column],2).abs()
    error.drop(columns=['HORA','bloque','HORA_ABS'],inplace=True)
    error=error.groupby(by=['start_date','semana_año','dia_semana']).agg('mean').reset_index()
    error.drop(columns=['start_date','dia_semana'],inplace=True)
    error=error.groupby(by=['semana_año']).agg('mean').round(2).reset_index()
    
    
    aux=error[['MAPE_'+str(i) for i in range(XGB_models)]].describe().transpose()
    aux = aux[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    
    
    
    file = open('save_model/tunning_parameters/param_tuning.txt', 'r') 
    list_dict = []
    for x in file:
        list_dict.append(ast.literal_eval(x.replace('\n','')))
    
    
    df_dict = pd.DataFrame.from_dict(list_dict, orient='columns')
    aux[df_dict.columns.tolist()] = df_dict.values
    
    aux = aux [['learning_rate',
           'max_depth', 'n_estimators', 'n_jobs', 'colsample_bytree',
           'colsample_bylevel','mean', 'std', 'min', '25%', '50%', '75%', 'max']].copy()
    
    print("For target: " +column)
    print ("Week :" + str(week))
    
    #save the best values
    best_values = list_dict[int(aux.sort_values(by=['mean']).index.tolist()[0].split("_")[-1])]
    print(best_values)
    with open('save_model/tunning_parameters/best_parameter_t_'+str(week)+'_target_'+str(target)+'.json', 'w') as fp:
        json.dump(best_values,fp)
    
    
    
    aux.to_csv(path+'resultados_tunning_target_'+str(target)+'_t_'+str(week)+'.csv', index=False, sep=';',decimal=',')
    return aux