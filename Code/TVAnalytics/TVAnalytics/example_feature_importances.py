# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 23:46:48 2021

@author: ignasi
"""

import os
import sys


import pandas as pd
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
module_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)

import TVAnalytics as TV
# Set your local data directory.
path = 'C:/Users/ignasi/Desktop/TVAnalytics'

os.chdir(path)






def plot_feature_importance(path, year=2020,week=35,target=0,time_step=1,agrupacion='all',n_features=10):
    
    """
    agrupacion: 'all','Agrupacion_numerica',
                'Agrupacion_categorical','Agrupacion_genero'     
   target: 0 o 1
   time_step: 1 al 6
   
   target 0 : indm25_64
target 1: indhmabc25_64
    
    """
    dict_target={
            0 : "indm25 64",1: "indhmabc25 64"
        }
    dict_agrupacion={
            'all' : "",'Agrupacion_numerica': " numéricas",
            'Agrupacion_categorical': ' categóricas',
            'Agrupacion_genero' : ' de género' ,
            'Descripción del contenido':' de descripción del contenido',
            'Programación': ' de programación' , 
            'Rating': ' de rating', 
            'Tandas': ' de tandas',
           'Tiempo': ' de tiempo'
        }
    df=pd.read_csv(path+str(year)+'_'+str(week)+'_t_'+str(time_step)+'_target_'+str(target)+"_summary_importance_columns_shap_target.csv")
    
    #Sort the DataFrame in order decreasing feature importance
    df.sort_values(by=['Importance'], ascending=False,inplace=True)
    if agrupacion!='all':
        df=df[df[agrupacion]==1].copy()
    
    df=df.iloc[:n_features,:].copy()
    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    palette = sns.color_palette("Blues_d",n_colors=15)
    palette.reverse()
    sns.barplot(x=df['Importance'], y=df['Variable'],palette=palette)#"#69d"  "Blues_r"
    #Add chart labels
    plt.title('Variables'+dict_agrupacion[agrupacion]+' más importantes para el target '+dict_target[target])
    plt.xlabel('Importancia')
    plt.ylabel('Variables')
    plt.savefig('Variables_'+agrupacion+'_target_'+dict_target[target]+'.png', bbox_inches='tight')
    plt.show()



if __name__ == "__main__":
    path="output/pred_for_last_week_data/shap_importance_columns/"
    year=2020
    week=35
    target=0
    time_step=1
    agrupacion='all'
    n_features=15
    
    for target in range(2):
        for agrupacion in ['all','Agrupacion_numerica','Agrupacion_categorical','Agrupacion_genero' ,'Descripción del contenido', 'Programación', 'Rating', 'Tandas',
       'Tiempo']:
            plot_feature_importance(path, year,week,target,time_step,agrupacion,n_features)
    
    