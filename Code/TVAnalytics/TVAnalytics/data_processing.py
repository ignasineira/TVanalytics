# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 19:13:59 2021

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
import multiprocessing

import holidays
#from tqdm import tqdm



def generate_data(start_year=2015,start_month=1,tunning=True,save_metadata=False,other_df=None,for_pred=False,change_data=False,codigo=np.nan):
    """"
    input:
        start_year:
        start_month:
        tunning: boolean, -true if for tunning parameters or train model for the last week
                            - false if you ingrese your own data
        save_metadata: boolena -when you do the tuning you have the option to save its importan  metadata like bloaue
        other_df: dataframe, if you ingrese your own data
        for_pred: boolean -True if you you want to save the metadata of the week to predict
        change_data: boolean deprecated
        codigo

    caso para nueva data
    #df_X=generate_data(start_year,start_month,tunning=True,other_df=pd.read_csv('input/X_test.csv',header=None))
    """


    if other_df is not None:
        df=other_df

    else:
        df = pd.read_csv('input\entrada_modelo_final.csv', sep=';', error_bad_lines=False, header=None)

    # add the name of the columns
    columns_name = pd.read_csv('input\columns_name_final.csv', sep=',').columns.tolist()
    df.columns = columns_name


    # Drop columns
    df.drop(columns=['semana', 'hora_minuto_inicio', 'hora_minuto_fin', 'temporada', 'capitulo', 'rostros',
                     'rostros_inv','tag'], inplace=True)

    df.drop(columns=['tag_t_'+str(i)for i in range(1,7)],inplace=True)

    #generate datetime variable
    df.start_date = pd.to_datetime(df.start_date)
    df=df[df['start_date'] >=str(datetime.date(year=start_year,month=start_month,day=1))].copy()

    df['dia_año'] = df.start_date.apply(lambda x: x.dayofyear)
    df['semana_año'] = df.start_date.apply(lambda x: x.weekofyear)

    #correcciones de la data
    df.loc[df.numero_tandas_bloque==0,'minutos_tandas_bloque']=0
    # Transform categorical data
    ## 1.for final data C13
    pass_to_category_bloque=True
    if pass_to_category_bloque:
        to_covert_cat = ['bloque']
        for col in to_covert_cat:
            df[col] = df[col].astype('category')
    ## 2. for each channel
    use_programa_data = True
    #to_covert_cat_1 = ['bloque_inicio','bloque_fin','genero','subgenero']
    to_covert_cat_1 = ['genero','subgenero']
    # Separate database by channel
    lista_canales = np.sort(df.canal.unique()).tolist()
    merge_list = ['año','mes','dia_mes','semana_año','semana_mes','dia_año','dia_mes','dia_semana','bloque']

    #sort values
    df=df.sort_values(by=['canal','start_date','bloque'], ignore_index=True)
    #list columns to predict
    lista_columns_pred = []
    with open("input\columns_predict.txt", "r") as f:
        for line in f:
            lista_columns_pred.append(line.strip())

    list_df_canales = []
    for canal in lista_canales:
        aux = df[df.canal == canal].copy()
        if canal == 5:
            #add holiday and weekend information
            chile_holiday=holidays.Chile(years=list(range(start_year-1, aux.año.max()+2)))
            list_holydays=sorted(list(chile_holiday.keys()))

            dates = np.sort(aux["start_date"].unique())
            #generate df of all days in the original data
            df_holidays = pd.DataFrame(dates, columns =['start_date'])
            #1 if that day is a holyday, 0 if not
            df_holidays['holiday']=np.where(df_holidays['start_date'].isin(list_holydays),1,0)
            #1 if that day+ tweek more is a holyday, 0 if not
            for t in range(1,7):
                df_holidays['holiday_t_'+str(t)] = np.where((df_holidays['start_date']+ pd.to_timedelta(7*t, unit="D")).isin(list_holydays),1,0)
            #1 if that day is a holyday or weekend, 0 if not
            df_holidays['holiday_weekend'] = np.where((df_holidays['start_date'].isin(list_holydays)) | (df_holidays['start_date'].dt.dayofweek==5) | (df_holidays['start_date'].dt.dayofweek==6),1,0)
            for t in range(1,7):
                df_holidays['holiday_weeken_t_'+str(t)] = np.where(((df_holidays['start_date']+ pd.to_timedelta(7*t, unit="D")).isin(list_holydays)) | (df_holidays['start_date'].dt.dayofweek==5) | (df_holidays['start_date'].dt.dayofweek==6),1,0)

            list_holydays_weekend = df_holidays[df_holidays['holiday_weekend']==1]['start_date'].unique()

            today=df_holidays['start_date'].iloc[-1]

            for t in range(0,7):
                offset = (today.weekday() + 4)%7+7*t
                next_saturday = today + timedelta(days=offset)
                list_holydays_weekend  = np.append(list_holydays_weekend,np.datetime64(next_saturday))

            list_holydays_weekend =pd.to_datetime([str(item) for item in list_holydays_weekend]).date


            df_holidays['days_next_holiday']= df_holidays.start_date.apply(lambda x: days_next_holiday(x, list_holydays))
            df_holidays['days_next_holiday_weekend']=df_holidays.start_date.apply(lambda x: days_next_holiday(x, list_holydays_weekend))
            for t in range(1,7):
                df_holidays['days_next_holiday_t_'+str(t)]= (df_holidays.start_date+ pd.to_timedelta(7*t, unit="D")).apply(lambda x: days_next_holiday(x, list_holydays))
                df_holidays['days_next_holiday_weekend_t_'+str(t)]=(df_holidays.start_date+ pd.to_timedelta(7*t, unit="D")).apply(lambda x: days_next_holiday(x, list_holydays_weekend))

            aux=pd.merge(aux,df_holidays,how='left',on=['start_date'])

            #add hour of the day, the filter by a condition
            bloque_hora=pd.read_csv('input/BLOQUES_ID_map.csv',sep=';')
            bloque_hora['HORA']=pd.to_datetime(bloque_hora['HORA']).dt.time
            bloque_hora['HORA']=bloque_hora['HORA'].apply(lambda t: t.strftime('%H:%M'))
            bloque_hora.rename(columns={'BLOQUES':'bloque'},inplace=True)
            aux=aux.merge(bloque_hora, on=['bloque'])
            aux.drop(columns=['HORA'],inplace=True)
            #eliminar datos de las 1 am hasta 8 am
            aux=aux[(aux.HORA_ABS<1) | (aux.HORA_ABS>=8)].copy()


            if change_data:
                #this function is deprecated, because se user will ingrese the data
                year_test=aux.año.max()
                week_test=aux[aux.año == year_test].semana_año.max()
                genero=aux[(aux.año==year_test)&(aux.semana_año==week_test)&(aux.codigo==codigo)].genero.unique()[0]
                subgenero=aux[(aux.año==year_test)&(aux.semana_año==week_test)&(aux.codigo==codigo)].subgenero.unique()[0]
                aux.loc[(aux.año==year_test)&(aux.semana_año==week_test),'genero']=genero
                aux.loc[(aux.año==year_test)&(aux.semana_año==week_test),'subgenero']=subgenero

            aux.sort_values(by=['start_date', 'HORA_ABS','bloque'],inplace=True, ignore_index=True)

            if tunning==True and save_metadata==True :
                aux[['codigo','start_date','año','mes','semana_año','semana_mes','dia_año','dia_mes','dia_semana','bloque','holiday','holiday_weekend','days_next_holiday','days_next_holiday_weekend']].to_csv('output/All_tunning_parameter/X_data_test.csv',index = False)
                #save data for  error analysis
                aux.to_csv('output/All_tunning_parameter/X_data_test_categorical_columns.csv',index = False)

            #save date time information
            aux = pd.get_dummies(aux, columns=['pais'])
            aux = pd.get_dummies(aux, columns=['genero_t_'+str(i)for i in range(1,7)])
            aux = pd.get_dummies(aux, columns=['subgenero_t_'+str(i)for i in range(1,7)])
            aux = pd.get_dummies(aux, columns=['pais_t_'+str(i)for i in range(1,7)])
            lista_aux=[]
            lista_aux.extend(['repeticion','resumen','lomejor','especial','envivo','serie'])
            for name in ['repeticion','resumen','lomejor','especial','envivo','serie']:
                lista_aux.extend([name+'_t_'+str(i)for i in range(1,7)])
            aux[lista_aux]=aux[lista_aux].replace({True: 1, False: 0})


            if for_pred:
                list_programa = []
                for lista in ['codigo','start_date','año','mes','semana_año','semana_mes','dia_año','dia_mes','dia_semana','bloque','holiday','holiday_weekend','days_next_holiday','days_next_holiday_weekend']:
                    list_programa.append(aux[lista].tolist())


        if canal != 5:
            #drop rating for the other channel beacuse at this moment that columns are empty
            aux.drop(columns=lista_columns_pred, inplace=True)
            aux.drop(columns=['pais'],inplace=True)
            aux.drop(columns=['genero_t_'+str(i)for i in range(1,7)],inplace=True)
            aux.drop(columns=['subgenero_t_'+str(i)for i in range(1,7)],inplace=True)
            aux.drop(columns=['pais_t_'+str(i)for i in range(1,7)],inplace=True)
            aux.drop(columns=['repeticion_t_'+str(i)for i in range(1,7)],inplace=True)
            aux.drop(columns=['resumen_t_'+str(i)for i in range(1,7)],inplace=True)
            aux.drop(columns=['lomejor_t_'+str(i)for i in range(1,7)],inplace=True)
            aux.drop(columns=['especial_t_'+str(i)for i in range(1,7)],inplace=True)
            aux.drop(columns=['envivo_t_'+str(i)for i in range(1,7)],inplace=True)
            aux.drop(columns=['serie_t_'+str(i)for i in range(1,7)],inplace=True)

        aux.drop(columns=['codigo','canal', 'start_date'], inplace=True)
        aux.drop(columns=['codigo_t_'+str(i)for i in range(1,7)],inplace=True)

        if use_programa_data:
            aux = pd.get_dummies(aux, columns=to_covert_cat_1)

        cols = aux.columns[~aux.columns.isin(merge_list)]
        aux.rename(columns=dict(zip(cols, cols + '_' + str(canal))), inplace=True)
        list_df_canales.append([aux, canal])

    # Merge data by canal
    id_C13=lista_canales.index(5)
    df_final = list_df_canales[id_C13][0]
    #print(df_final.shape)
    if len(list_df_canales)>1:
        
        for df_canal in list_df_canales:
            if df_canal[1]!=5:
                df_canal[0]=df_canal[0].drop_duplicates(subset=merge_list)
                df_final = df_final.merge(df_canal[0], on=merge_list, how='left')
                #print(df_canal[1],df_final.shape)
    # Transform to dummy columns
    if pass_to_category_bloque:
        df_final = pd.get_dummies(df_final, columns=to_covert_cat)
    df_final = df_final.reset_index(drop=True)

    df_final.fillna(np.nan, inplace=True)
    obj_columns = list(df_final.select_dtypes(include=['object']).columns.values)
    df_final[obj_columns] = df_final[obj_columns].replace([None], np.nan)
    #df_final.replace('None', np.nan, inplace=True)
    if tunning:
        pd.DataFrame(columns=df_final.columns.tolist()).to_csv('input/columns_name_post_pipeline.csv',index = False)

    elif tunning==False:
        # Get missing columns in the training test
        train_columns=pd.read_csv('input\columns_name_post_pipeline.csv', sep=',').columns.tolist()
        missing_cols = set(train_columns) - set( df_final.columns )
        # Add a missing column in test set with default value equal to 0
        for c in missing_cols:
            df_final[c] = 0
        # Ensure the order of column in the test set is in the same order than in train set
        #and delete the columns with which the model never trained
        df_final = df_final[train_columns]

    if for_pred:
        return df_final,list_programa

    elif for_pred==False:

        return df_final


def generate_data_time_slot(start_year=2015, start_month=1, tunning=True, save_metadata=False, other_df=None, for_pred=False,
                  change_data=False, codigo=np.nan):
    """"
    input:
        start_year:
        start_month:
        tunning: boolean, -true if for tunning parameters or train model for the last week
                            - false if you ingrese your own data
        save_metadata: boolena -when you do the tuning you have the option to save its importan  metadata like bloaue
        other_df: dataframe, if you ingrese your own data
        for_pred: boolean -True if you you want to save the metadata of the week to predict
        change_data: boolean deprecated
        codigo

    caso para nueva data
    #df_X=generate_data(start_year,start_month,tunning=True,other_df=pd.read_csv('input/X_test.csv',header=None))
    """

    if other_df is not None:
        df = other_df

    else:
        df = pd.read_csv('input\entrada_modelo_final.csv', sep=';', error_bad_lines=False, header=None)

    # add the name of the columns
    columns_name = pd.read_csv('input\columns_name_final.csv', sep=',').columns.tolist()
    df.columns = columns_name

    # Drop columns
    df.drop(columns=['semana', 'hora_minuto_inicio', 'hora_minuto_fin', 'temporada', 'capitulo', 'rostros',
                     'rostros_inv', 'tag'], inplace=True)

    df.drop(columns=['tag_t_' + str(i) for i in range(1, 7)], inplace=True)

    # generate datetime variable
    df.start_date = pd.to_datetime(df.start_date)
    df = df[df['start_date'] >= str(datetime.date(year=start_year, month=start_month, day=1))].copy()

    df['dia_año'] = df.start_date.apply(lambda x: x.dayofyear)
    df['semana_año'] = df.start_date.apply(lambda x: x.weekofyear)

    # correcciones de la data
    df.loc[df.numero_tandas_bloque == 0, 'minutos_tandas_bloque'] = 0
    # Transform categorical data
    ## 1.for final data C13
    pass_to_category_bloque = True
    if pass_to_category_bloque:
        to_covert_cat = ['bloque']
        for col in to_covert_cat:
            df[col] = df[col].astype('category')
    ## 2. for each channel
    use_programa_data = True
    # to_covert_cat_1 = ['bloque_inicio','bloque_fin','genero','subgenero']
    to_covert_cat_1 = ['genero', 'subgenero']
    # Separate database by channel
    lista_canales = df.canal.unique()
    merge_list = ['año', 'mes', 'dia_mes', 'semana_año', 'semana_mes', 'dia_año', 'dia_mes', 'dia_semana', 'bloque']

    # sort values
    df = df.sort_values(by=['canal', 'start_date', 'bloque'], ignore_index=True)
    # list columns to predict
    lista_columns_pred = []
    with open("input\columns_predict.txt", "r") as f:
        for line in f:
            lista_columns_pred.append(line.strip())
    
    
    chile_holiday = holidays.Chile(years=list(range(start_year - 1, df.año.max() + 2)))
    list_holydays = sorted(list(chile_holiday.keys()))

    dates = np.sort(df["start_date"].unique())
    # generate df of all days in the original data
    df_holidays = pd.DataFrame(dates, columns=['start_date'])
    # 1 if that day is a holyday, 0 if not
    df_holidays['holiday'] = np.where(df_holidays['start_date'].isin(list_holydays), 1, 0)
    # 1 if that day+ tweek more is a holyday, 0 if not
    for t in range(1, 7):
        df_holidays['holiday_t_' + str(t)] = np.where(
            (df_holidays['start_date'] + pd.to_timedelta(7 * t, unit="D")).isin(list_holydays), 1, 0)
    # 1 if that day is a holyday or weekend, 0 if not
    df_holidays['holiday_weekend'] = np.where(
        (df_holidays['start_date'].isin(list_holydays)) | (df_holidays['start_date'].dt.dayofweek == 5) | (
                    df_holidays['start_date'].dt.dayofweek == 6), 1, 0)
    for t in range(1, 7):
        df_holidays['holiday_weekend_t_' + str(t)] = np.where(
            ((df_holidays['start_date'] + pd.to_timedelta(7 * t, unit="D")).isin(list_holydays)) | (
                        df_holidays['start_date'].dt.dayofweek == 5) | (
                        df_holidays['start_date'].dt.dayofweek == 6), 1, 0)

    list_holydays_weekend = df_holidays[df_holidays['holiday_weekend'] == 1]['start_date'].unique()

    today = df_holidays['start_date'].iloc[-1]

    for t in range(0, 7):
        offset = (today.weekday() + 4) % 7 + 7 * t
        next_saturday = today + timedelta(days=offset)
        list_holydays_weekend = np.append(list_holydays_weekend, np.datetime64(next_saturday))

    list_holydays_weekend = pd.to_datetime([str(item) for item in list_holydays_weekend]).date

    df_holidays['days_next_holiday'] = df_holidays.start_date.apply(
        lambda x: days_next_holiday(x, list_holydays))
    df_holidays['days_next_holiday_weekend'] = df_holidays.start_date.apply(
        lambda x: days_next_holiday(x, list_holydays_weekend))
    for t in range(1, 7):
        df_holidays['days_next_holiday_t_' + str(t)] = (
                    df_holidays.start_date + pd.to_timedelta(7 * t, unit="D")).apply(
            lambda x: days_next_holiday(x, list_holydays))
        df_holidays['days_next_holiday_weekend_t_' + str(t)] = (
                    df_holidays.start_date + pd.to_timedelta(7 * t, unit="D")).apply(
            lambda x: days_next_holiday(x, list_holydays_weekend))
    
    
    bloque_hora = pd.read_csv('input/BLOQUES_ID_map.csv', sep=';')
    franja = pd.read_csv('input/HORA_FRANJA_ID_map.csv', sep=';')
    bloque_hora['HORA'] = pd.to_datetime(bloque_hora['HORA']).dt.time
    bloque_hora['HORA'] = bloque_hora['HORA'].apply(lambda t: t.strftime('%H:%M'))
    bloque_hora.rename(columns={'BLOQUES': 'bloque'}, inplace=True)
    bloque_hora=pd.merge(bloque_hora,franja,how='inner', on=['HORA_ABS'])
    
    aux_franja=bloque_hora[['bloque', 'time_slot']].copy()
    df=pd.merge(df,aux_franja,how='inner', on=['bloque'])
    
    dict_df_final={}
    dict_list_programa={}
    for i, (group_name, group_df) in enumerate(df.groupby(["time_slot"])):
        list_df_canales = []
        for canal in lista_canales:
            aux = group_df[group_df.canal == canal].copy()
            aux.drop(columns=["time_slot"], inplace= True)
            if canal == 5:
                # add holiday and weekend information
                aux = pd.merge(aux, df_holidays, how='left', on=['start_date'])
    
                # add hour of the day, the filter by a condition
                bloque_hora = pd.read_csv('input/BLOQUES_ID_map.csv', sep=';')
                #franja = pd.read_csv('input/HORA_FRANJA_ID_map.csv', sep=';')
                bloque_hora['HORA'] = pd.to_datetime(bloque_hora['HORA']).dt.time
                bloque_hora['HORA'] = bloque_hora['HORA'].apply(lambda t: t.strftime('%H:%M'))
                bloque_hora.rename(columns={'BLOQUES': 'bloque'}, inplace=True)
                #bloque_hora=pd.merge(bloque_hora,franja,how='inner', on=['HORA_ABS'])
                aux = aux.merge(bloque_hora, on=['bloque'])
                aux.drop(columns=['HORA'], inplace=True)
                # eliminar datos de las 1 am hasta 8 am
                aux = aux[(aux.HORA_ABS < 1) | (aux.HORA_ABS >= 8)].copy()
    
                if change_data:
                    # this function is deprecated, because se user will ingrese the data
                    year_test = aux.año.max()
                    week_test = aux[aux.año == year_test].semana_año.max()
                    genero = \
                    aux[(aux.año == year_test) & (aux.semana_año == week_test) & (aux.codigo == codigo)].genero.unique()[0]
                    subgenero = \
                    aux[(aux.año == year_test) & (aux.semana_año == week_test) & (aux.codigo == codigo)].subgenero.unique()[
                        0]
                    aux.loc[(aux.año == year_test) & (aux.semana_año == week_test), 'genero'] = genero
                    aux.loc[(aux.año == year_test) & (aux.semana_año == week_test), 'subgenero'] = subgenero
    
                aux.sort_values(by=['start_date', 'HORA_ABS', 'bloque'], inplace=True, ignore_index=True)
    
                if tunning == True and save_metadata == True:
                    aux[['codigo', 'start_date', 'año', 'mes', 'semana_año', 'semana_mes', 'dia_año', 'dia_mes',
                         'dia_semana', 'bloque', 'holiday', 'holiday_weekend', 'days_next_holiday',
                         'days_next_holiday_weekend']].to_csv('output/All_tunning_parameter_time_slot/X_data_test_'+group_name+'.csv', index=False)
                    # save data for  error analysis
                    aux.to_csv('output/All_tunning_parameter_time_slot/X_data_test_categorical_columns_'+group_name+'.csv', index=False)
    
                # save date time information
                aux = pd.get_dummies(aux, columns=['pais'])
                aux = pd.get_dummies(aux, columns=['genero_t_' + str(i) for i in range(1, 7)])
                aux = pd.get_dummies(aux, columns=['subgenero_t_' + str(i) for i in range(1, 7)])
                aux = pd.get_dummies(aux, columns=['pais_t_' + str(i) for i in range(1, 7)])
                lista_aux = []
                lista_aux.extend(['repeticion', 'resumen', 'lomejor', 'especial', 'envivo', 'serie'])
                for name in ['repeticion', 'resumen', 'lomejor', 'especial', 'envivo', 'serie']:
                    lista_aux.extend([name + '_t_' + str(i) for i in range(1, 7)])
                aux[lista_aux] = aux[lista_aux].replace({True: 1, False: 0})
    
                if for_pred:
                    list_programa = []
                    for lista in ['codigo', 'start_date', 'año', 'mes', 'semana_año', 'semana_mes', 'dia_año', 'dia_mes',
                                  'dia_semana', 'bloque', 'holiday', 'holiday_weekend', 'days_next_holiday',
                                  'days_next_holiday_weekend']:
                        list_programa.append(aux[lista].tolist())
                    dict_list_programa[group_name]=list_programa
            if canal != 5:
                # drop rating for the other channel beacuse at this moment that columns are empty
                aux.drop(columns=lista_columns_pred, inplace=True)
                aux.drop(columns=['pais'], inplace=True)
                aux.drop(columns=['genero_t_' + str(i) for i in range(1, 7)], inplace=True)
                aux.drop(columns=['subgenero_t_' + str(i) for i in range(1, 7)], inplace=True)
                aux.drop(columns=['pais_t_' + str(i) for i in range(1, 7)], inplace=True)
                aux.drop(columns=['repeticion_t_' + str(i) for i in range(1, 7)], inplace=True)
                aux.drop(columns=['resumen_t_' + str(i) for i in range(1, 7)], inplace=True)
                aux.drop(columns=['lomejor_t_' + str(i) for i in range(1, 7)], inplace=True)
                aux.drop(columns=['especial_t_' + str(i) for i in range(1, 7)], inplace=True)
                aux.drop(columns=['envivo_t_' + str(i) for i in range(1, 7)], inplace=True)
                aux.drop(columns=['serie_t_' + str(i) for i in range(1, 7)], inplace=True)

            aux.drop(columns=['codigo', 'canal', 'start_date'], inplace=True)
            aux.drop(columns=['codigo_t_' + str(i) for i in range(1, 7)], inplace=True)
    
            if use_programa_data:
                aux = pd.get_dummies(aux, columns=to_covert_cat_1)
    
            cols = aux.columns[~aux.columns.isin(merge_list)]
            aux.rename(columns=dict(zip(cols, cols + '_' + str(canal))), inplace=True)
            list_df_canales.append([aux, canal])

        # Merge data by canal
        df_final = list_df_canales[3][0]
        # print(df_final.shape)
        for df_canal in list_df_canales[:-1]:
            df_canal[0] = df_canal[0].drop_duplicates(subset=merge_list)
            df_final = df_final.merge(df_canal[0], on=merge_list, how='left')
        # print(df_canal[1],df_final.shape)
        # Transform to dummy columns
        if pass_to_category_bloque:
            df_final = pd.get_dummies(df_final, columns=to_covert_cat)
        df_final = df_final.reset_index(drop=True)

        df_final.fillna(np.nan, inplace=True)
        if tunning:
            pd.DataFrame(columns=df_final.columns.tolist()).to_csv('input/columns_name_post_pipeline_'+group_name+'.csv', index=False)

        elif tunning == False:
            # Get missing columns in the training test
            train_columns = pd.read_csv('input\columns_name_post_pipeline'+group_name+'.csv', sep=',').columns.tolist()
            missing_cols = set(train_columns) - set(df_final.columns)
            # Add a missing column in test set with default value equal to 0
            for c in missing_cols:
                df_final[c] = 0
            # Ensure the order of column in the test set is in the same order than in train set
            # and delete the columns with which the model never trained
            df_final = df_final[train_columns]
        dict_df_final[group_name]=df_final
    
    if for_pred:
        return dict_df_final, dict_list_programa

    elif for_pred == False:

        return dict_df_final


def days_next_holiday(date, holidays):
    difference=[]
    for item in holidays:
        difference.append((item-date.date()).days)
        
    return min([x for x in difference if x>=0])