# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 22:21:23 2021

@author: ignasi
"""

import os 
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt

import plotly.express as px

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import plotly.offline as py
#py.plot(fig, filename='image/graph.png')
pd.set_option("display.precision", 8)





def main_plot_tunning():
    
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
            
            
            
            path_image='image/All_tunning_parameter/week_'+str(week)+"/"
    
            error=df.copy()
            XGB_models = sum('XGB_' in s for s in df.columns) 
            for i in range(XGB_models):
                error['MAPE_'+str(i)]=round(100*(error[column]-error['XGB_'+str(i)])/error[column],2).abs()
            error.drop(columns=['HORA','bloque','HORA_ABS'],inplace=True)
            error=error.groupby(by=['start_date','semana_año','dia_semana']).agg('mean').reset_index()
            error.drop(columns=['start_date','dia_semana'],inplace=True)
            error=error.groupby(by=['semana_año']).agg('mean').round(2).reset_index()
            
            aux = error[['MAPE_'+str(i) for i in range(XGB_models)]]
            aux.columns= ['Model '+str(i) for i in range(XGB_models)]
            aux = aux.describe().transpose()
            aux = aux[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
            aux=aux.sort_values(by=['mean'])
            columns_to_plot = aux.index.tolist()[0:: (len(aux)-1)//2]
            
            lista_1 = ['MAPE_'+str(i) for i in range(XGB_models)]#.append('semana_año')
            lista_1.append('semana_año')
            ax = error[lista_1].sort_values(by=['semana_año'])
            lista_2 = ['Model '+str(i) for i in range(XGB_models)]#.append('semana_año')
            lista_2.append('semana_año')
            ax.columns = lista_2 
            fig = px.line(ax,x='semana_año',y=columns_to_plot)
            
            fig.update_layout(
                autosize=False,
                width=1200,
                height=800,
                title="Evolución MAPE agrupado por semana año 2020",
                xaxis_title="Semana del año",
                yaxis_title="MAPE",
                legend_title="Modelos",
                
                font=dict(
                    #family="Courier New, monospace",
                    size=18,
                    color="Black"
                )
            )
            #fig.update_xaxes(rangeslider_visible=True)
            fig.write_image(path_image+"target_"+column+"_MAPE_por_semana_año.png")
            
            #serie de tiempo agrupada por bloque
            lista_1 = ['MAPE_'+str(i) for i in range(XGB_models)]#.append('semana_año')
            lista_1.append('HORA')
            error=df.copy()
            for i in range(XGB_models):
                error['MAPE_'+str(i)]=round(100*(error[column]-error['XGB_'+str(i)])/error[column],2).abs()
            error=error[lista_1].copy()
            error=error.groupby(by=['HORA']).agg('mean').reset_index()
            #error.drop(columns=['start_date','dia_semana'],inplace=True)
            #error=error.groupby(by=['DIA','BLOQUES','HORA']).agg('mean').reset_index()
            #error.head()
            
            ax = error[lista_1].sort_values(by=['HORA'])
            lista_2 = ['Model '+str(i) for i in range(XGB_models)]#.append('semana_año')
            lista_2.append('HORA')
            ax.columns = lista_2
            fig = px.line(ax,x='HORA',y=columns_to_plot,
                         width=1200, height=1000)
            
            fig.update_layout(
                title="MAPE agrupado bloque",
                xaxis_title="Hora",
                yaxis_title="MAPE",
                legend_title="Variables",
                font=dict(
                    family="Courier New, monospace",
                    size=18,
                    color="RebeccaPurple"
                )
            )
            fig.write_image(path_image+"target_"+column+"_MAPE_por_bloque.png")
            
            #serie de tiempo agrupada por dia
            lista_1 = ['MAPE_'+str(i) for i in range(XGB_models)]#.append('semana_año')
            lista_1.append('dia_semana')
            error=df.copy()
            for i in range(XGB_models):
                error['MAPE_'+str(i)]=round(100*(error[column]-error['XGB_'+str(i)])/error[column],2).abs()
            error=error[lista_1].copy()
            error=error.groupby(by=['dia_semana']).agg('mean').reset_index()
            #error.drop(columns=['start_date','dia_semana'],inplace=True)
            #error=error.groupby(by=['DIA','BLOQUES','HORA']).agg('mean').reset_index()
            #error.head()
            
            ax = error[lista_1].sort_values(by=['dia_semana'])
            lista_2 = ['Model '+str(i) for i in range(XGB_models)]#.append('semana_año')
            lista_2.append('dia_semana')
            ax.columns = lista_2
            fig = px.line(ax,x='dia_semana',y=columns_to_plot,
                         width=1200, height=1000)
            
            fig.update_layout(
                title="MAPE agrupado dia de la semana",
                xaxis_title="Día de la semana",
                yaxis_title="MAPE",
                legend_title="Variables",
                font=dict(
                    family="Courier New, monospace",
                    size=18,
                    color="RebeccaPurple"
                )
            )
            fig.write_image(path_image+"target_"+column+"_MAPE_por_dia.png")
            
            #serie de tiempo agrupada por Dia-bloque
            lista_1 = ['MAPE_'+str(i) for i in range(XGB_models)]#.append('semana_año')
            lista_1.append('dia_semana')
            lista_1.append('HORA')
            error=df.copy()
            for i in range(XGB_models):
                error['MAPE_'+str(i)]=round(100*(error[column]-error['XGB_'+str(i)])/error[column],2).abs()
            error=error[lista_1].copy()
            error=error.groupby(by=['dia_semana','HORA']).agg('mean').reset_index()
            ax = error[lista_1].sort_values(by=['dia_semana','HORA'])
            lista_2 = ['Model '+str(i) for i in range(XGB_models)]#.append('semana_año')
            lista_2.append('dia_semana')
            lista_2.append('HORA')
            ax.columns = lista_2
            fig = px.line(ax,x='HORA',y=columns_to_plot,facet_row='dia_semana',
             width=1200, height=2000)
            
            fig.update_layout(
                title="MAPE agrupado dia de la semana-bloque",
                xaxis_title="Hora",
                yaxis_title="MAPE",
                legend_title="Variables",
                font=dict(
                    family="Courier New, monospace",
                    size=18,
                    color="RebeccaPurple"
                )
            )
            #fig.show()            
            fig.write_image(path_image+"target_"+column+"_MAPE_por_dia_bloque.png")
            
            

            
