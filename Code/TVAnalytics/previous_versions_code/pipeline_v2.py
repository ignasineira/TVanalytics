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
import holidays
#from tqdm import tqdm


def pipeline_tunning_param(n_week_train=15,start_year=2015,start_month=1,generate_new_df=False):
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
        df_final=generate_data(start_year,start_month)
        df_final.to_csv('input/df_final.csv',index = False)
    else:
        df_final=pd.read_csv('input/df_final.csv')
    fin= timer()-inicio
    print('Data ready! time(s): ',round(fin,2))
    lista_columns=[s+'_5' for s in lista_columns]
    
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


    param_tuning = {
        'learning_rate': [0.02],
        'max_depth': [5,6,7],
        'n_estimators': [50,100,150,200,500],
        'n_jobs': [multiprocessing.cpu_count() // 2],
        'colsample_bytree': [0.1,0.5,1],
        'colsample_bylevel':[0.5,1],
        #'colsample_bynode ':[0.5,1],#no se usa
        'seed': [1]
    }
    
    
    keys, values = zip(*param_tuning.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    #save the all the combinations in a list
    with open('save_model/tunning_parameters/param_tuning.txt', 'w') as f:
        for item in permutations_dicts:
            f.write("%s\n" % item)
    num_model=len(permutations_dicts)
    print('Time to train model!')
    inicio=timer()
    train_model(df_X,df_Y,num_model,permutations_dicts,list_tuple_week,index_year_week_train,list_index,preprocessing=0)
    fin= (timer()-inicio)/60
    print('Predict ready! time(m): ',round(fin,2))


def generate_data(start_year=2015,start_month=1,tunning=True,other_df=None,change_data=False,codigo=np.nan):
    """"
    
    caso para nueva data
    #df_X=generate_data(start_year,start_month,tunning=True,other_df=pd.read_csv('input/X_test.csv',header=None))
    """
    if tunning:
        df = pd.read_csv('input\entrada_modelo_final.csv', sep=';', error_bad_lines=False, header=None)
    
    elif tunning==False:
        df=other_df
        
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
    to_covert_cat = ['bloque']
    for col in to_covert_cat:
        df[col] = df[col].astype('category')
    ## 2. for each channel
    use_programa_data = True
    to_covert_cat_1 = ['bloque_inicio','bloque_fin','genero','subgenero']

    # Separate database by channel
    lista_canales = df.canal.unique()
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
                year_test=aux.año.max()
                week_test=aux[aux.año == year_test].semana_año.max()
                genero=aux[(aux.año==year_test)&(aux.semana_año==week_test)&(aux.codigo==codigo)].genero.unique()[0]
                subgenero=aux[(aux.año==year_test)&(aux.semana_año==week_test)&(aux.codigo==codigo)].subgenero.unique()[0]
                aux.loc[(aux.año==year_test)&(aux.semana_año==week_test),'genero']=genero
                aux.loc[(aux.año==year_test)&(aux.semana_año==week_test),'subgenero']=subgenero
                
            aux.sort_values(by=['start_date', 'HORA_ABS','bloque'],inplace=True, ignore_index=True)    
            
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
            
            #list_programa = []
            #for lista in ['codigo','start_date','año','mes','semana_año','semana_mes','dia_año','dia_mes','dia_semana','bloque','holiday','holiday_weekend','days_next_holiday','days_next_holiday_weekend']:
                #list_programa.append(aux[lista].tolist())
            if tunning:
                aux[['codigo','start_date','año','mes','semana_año','semana_mes','dia_año','dia_mes','dia_semana','bloque','holiday','holiday_weekend','days_next_holiday','days_next_holiday_weekend']].to_csv('output/All_tunning_parameter/X_data_test.csv',index = False)  
            
        if canal != 5:
            #drop rating for the other channel
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
    df_final = list_df_canales[3][0]
    #print(df_final.shape)
    for df_canal in list_df_canales[:-1]:
        df_canal[0]=df_canal[0].drop_duplicates(subset=merge_list)
        df_final = df_final.merge(df_canal[0], on=merge_list, how='left')
        #print(df_canal[1],df_final.shape)
    # Transform to dummy columns
    df_final = pd.get_dummies(df_final, columns=to_covert_cat)
    df_final = df_final.reset_index(drop=True)
    
    df_final.fillna(np.nan, inplace=True)
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
        df_final = df_final[train_columns]
            

    return df_final


def train_model(df_X,df_Y,num_model,permutations_dicts,list_tuple_week,index_year_week_train,list_index,preprocessing=0):
    preprocessing=str(preprocessing)
    dic_aux={0:(1,0),1:(1,1),2:(2,0),3:(2,1),4:(3,0),5:(3,1),6:(4,0),7:(4,1),8:(5,0),9:(5,1),10:(6,0),11:(6,1)}
    key_list = list(dic_aux.keys())
    val_list = list(dic_aux.values())
    
    list_columns_always_use=[col for col in df_X.columns if '_t_' not in col]
    list_columns_always_use_location=[df_X.columns.get_loc(c) for c in list_columns_always_use ]
    for t in range(1, 7):
    #for t in range(1, 2):
        print("Time step predict: ", t )
        print('**--------------------------------------------**')
        print(' ')
        # Run sliding window regression models
        for target in range(2):
            
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
                    reg = XGBRegressor(**permutations_dicts[j]).fit(X_train_i, Y_train_i)
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
                Y_test_i.to_csv("output\All_tunning_parameter\week_"+str(t)+"\model_predict_xgb_tunning_target_"+str(target)+"_pre_"+preprocessing+"_year_"+str(list_tuple_week[i + 1][0])+"_week_"+str(list_tuple_week[i + 1][1])+".csv")
                
                print("READY WEEK!")
                print(lista_errores_semana)
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


def days_next_holiday(date, holidays):
    difference=[]
    for item in holidays:
        difference.append((item-date.date()).days)
        
    return min([x for x in difference if x>=0])




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
    df_final,list_programa=generate_data(start_year,start_month)
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
    for t in range(1, 7):
    #for t in range(2, 7):
        print("Time step predict: ", t )
        print('**--------------------------------------------**')
        print(' ')
        # Run sliding window regression models
        for target in range(2):
        #for target in range(1):
            
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
            
            lo = list_index[i] + 1
            lo_test = list_index[j] + 1
            hi = list_index[j + 1] + 1
            X_train_i = df_X.iloc[:lo, :]
            Y_train_i = df_Y.iloc[:lo, key_list[position]]
            X_test_i = df_X.iloc[lo_test:, :]
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
    
            Y_test.to_csv("output\pred_for_last_week_data\week_"+str(t)+"/model_predict_xgb_tunning_"+str(year_test)+"_"+str(week_test)+"_target_"+str(target)+".csv")
            sys.stdout.write(f'Total time for target {target} and Time step {t} (m):{fin_model/60:02.3f} .')
            print(' ')
            print('##----------------------####-------------------------##')
    
    fin= (timer()-inicio)/60
    print('Predict ready! time(m): ',round(fin,2))



def pred_changing_data(start_year=2015,start_month=1,codigo=46892):
    """
    esta función está obsoleta, porque el usuario ingresará los datos
    
    """
    #[ 9399,  3211, 33019, 46892, 28502,45917, 46017]
    #['TELETRECE A.M.','TELETRECE TARDE','T13 CIUDADANOS','TELETRECE','TELETRECE A LA HORA','BIENVENIDOS CADA UNO CUENTA','BIENVENIDOS CADA UNO CUENTA']
    #list of columns predict
    inicio=timer()
    lista_columns = []
    with open("input\columns_predict.txt", "r") as f:
        for line in f:
            lista_columns.append(line.strip())
    df_final,list_programa=generate_data(start_year=2015,start_month=1,change_data=True,codigo=codigo)
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
            
    print('Time to test model changing data!')
    inicio=timer()

    key_list = list(dic_aux.keys())
    val_list = list(dic_aux.values())
    for t in range(1, 7):
    #for t in range(2, 7):
        print("Time step predict: ", t )
        print('**--------------------------------------------**')
        print(' ')
        # Run sliding window regression models
        for target in range(2):
        #for target in range(1):
            
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
            
            lo = list_index[i] + 1
            lo_test = list_index[j] + 1
            hi = list_index[j + 1] + 1
            X_train_i = df_X.iloc[:lo, :]
            Y_train_i = df_Y.iloc[:lo, key_list[position]]
            X_test_i = df_X.iloc[lo_test:, :]
            print()
            ini_model=timer()
            
            lista_errores_semana=[]
            
            reg = XGBRegressor()
            reg.load_model('save_model/last_model/last_model_t_'+str(t)+'_target_'+str(target)+'.model')
            lista_errores_semana.append(((Y_train_i - reg.predict(X_train_i)) ** 2).mean())
            MAPE_train=MAPE(Y_train_i,reg.predict(X_train_i))
            MAPE_train5000=MAPE(Y_train_i,reg.predict(X_train_i),5000)
            Yhat_test_all_XGB[lo - index_year_week_train - 1:hi - index_year_week_train - 1, 0] = reg.predict(
                            X_test_i).reshape((-1))
            
            fin_model=timer()-ini_model
            
            print(' ')
            sys.stdout.write(f'MSE train:{lista_errores_semana[-1]:02.3f}, '
                             +f', MAPE train:{MAPE_train:02.3f}%, '
                             +f',sin threshold MAPE train:{MAPE_train5000:02.3f}%, '
                             +f', Time train:{fin_model:02.3f} s.')
            
            
                
            print("READY WEEK!")
            print(' ')
    
            Y_test=pd.DataFrame(Y_test) #suppose is null
            
            Y_test.loc[:, "XGB_t_" + str(t)+'_target_'+ str(target)] = Yhat_test_all_XGB[:, 0]
    
            lista_aux=['codigo','start_date', 'año', 'mes', 'semana_año', 'semana_mes', 'dia_año', 'dia_mes', 'dia_semana', 'bloque']
            for i in range(len(lista_aux)):
                Y_test[lista_aux[i]] = list_programa[i][index_year_week_train + 1:]
            
            year_test=df_final.año.max()
            week_test=df_final[df_final.año == year_test].semana_año.max()
            Y_test.to_csv("output\pred_changing_data\week_"+str(t)+"/model_predict_xgb_tunning_"+str(year_test)+"_"+str(week_test)+"_target_"+str(target)+"_codido_"+str(codigo)+".csv")
            sys.stdout.write(f'Total time for target {target} and Time step {t} (m):{fin_model/60:02.3f} .')
            print(' ')
            print('##----------------------####-------------------------##')
    
    fin= (timer()-inicio)/60
    print('Predict ready! time(m): ',round(fin,2))



def generate_new_directory(path, include_week=True):
    if include_week:
        for i in range(1,7):
            try:
                os.makedirs(path+'\week_'+str(i))
            except OSError:
                print()
    else:
        try:
            os.makedirs(path)
        except OSError:
            print()
                 
            
            
def train_model_time_step_target(df_X,df_Y,num_model,permutations_dicts,list_tuple_week,index_year_week_train,list_index,t,target,preprocessing=0):
    pass
