# -*- coding: utf-8 -*-
"""
Created on Sun May  2 19:36:48 2021

@author: ignasi
"""

import warnings
warnings.filterwarnings('ignore')

# Load modules.
import os
import sys
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from xgboost import XGBRegressor
import xgboost as xgb
#import multiprocessing
import shap

import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from matplotlib import pyplot
from xgboost import plot_importance, plot_tree

def plot_importance_columns_last_week(reg,X_test_i,year_test,week_test,t,target):
    
    canal_dict={'1': 'CHV','3':'MEGA','4':'TVN','5':'C13'}
    
    explainer = shap.TreeExplainer(reg)
    shap_values = explainer.shap_values(X_test_i)
    
    """
    Add new category
    """
    aux_name_columns_df=pd.read_csv('input/columns_tipo_de_información.csv')
    
    aux_name_columns_df=pd.merge(pd.DataFrame( X_test_i.columns.tolist(), columns=['Variable']),aux_name_columns_df,how='left',on=['Variable'] )
    #['Variable', 'Tipo de información']
    all_columns= [(column[:-1]+canal_dict[column[-1]]).replace("_"," ") if (column[-1].isnumeric() and column[-2]=='_') else column.replace("_"," ") for column in  X_test_i.columns.tolist()]
    all_columns=[column.replace("holiday","feriado").replace("weekend ","fin de semana ").replace("weeken ","fin de semana ").replace("avg","promedio").replace("next","proximo").lower() for column in all_columns ]
    
    aux_tipo_de_información=aux_name_columns_df['Tipo de información'].tolist()
    list_categorical_name=['pais', 'genero','subgenero','repeticion','resumen','lomejor', 'especial','envivo','serie', 'bloque','P1','P2','P3','P4','P5','P6','P7','holiday weekend','holiday','feriado','fin de semana']
    Categorical_columns = []
    for column in all_columns:
        if any(word in column for word in list_categorical_name):
            if not any(word in column for word in ['next','avg', 'anterior','siguiente','contenido','inicio','fin','indhmabc25','indm25', 'promedio', 'proximo']):
                Categorical_columns.append(column)
                
    
    Categorical_columns_index=[all_columns.index(word) for word in Categorical_columns]
    Numeric_columns = [column for column in all_columns if (column not in Categorical_columns)]
    Numeric_columns_index=[all_columns.index(word) for word in Numeric_columns]
    
    
    Genero_columns= [column for column in all_columns if ('genero' in column)]
    Genero_columns_index=[all_columns.index(word) for word in Genero_columns]
    
    max_display= len(X_test_i.columns)
    
    feature_order = np.argsort(np.sum(np.abs(shap_values),axis=0))
    feature_order = feature_order[-min(max_display,len(feature_order)):]
    feature_inds = feature_order[:max_display]
    df_shap_values =np.abs(shap_values).mean(0)
            
    shap_importance = pd.DataFrame(data=df_shap_values[feature_inds], index=[all_columns[i] for i in feature_inds],
                                           columns=['Importance'])
    
    shap_importance['Agrupacion_numerica']=[1 if all_columns[i] in Numeric_columns else 0 for i in feature_inds ]
    shap_importance['Agrupacion_categorical']=[1 if all_columns[i] in Categorical_columns else 0 for i in feature_inds ]
    shap_importance['Agrupacion_genero']=[1 if all_columns[i] in Genero_columns else 0 for i in feature_inds ]
    shap_importance['Tipo de información']=[aux_tipo_de_información[i] for i in feature_inds]
    shap_importance= pd.get_dummies(shap_importance, columns=['Tipo de información'], prefix='', prefix_sep='')
    
    shap_importance.reset_index(inplace=True)
    shap_importance.rename(columns = {'index':'Variable'}, inplace=True)
    shap_importance.to_csv("output/pred_for_last_week_data/shap_importance_columns/"+str(year_test)+'_'+str(week_test)+'_t_'+str(t)+'_target_'+str(target)+"_summary_importance_columns_shap_target.csv",index=False)
    reg.get_booster().feature_names = all_columns
    #plot all the columns
    shap.summary_plot(explainer.shap_values(X_test_i.iloc[:,:]),
                  features = X_test_i.iloc[:,:],
                  feature_names=all_columns,show=False) 
    plt.savefig("image/plot_importance_columns/"+str(year_test)+'_'+str(week_test)+'_t_'+str(t)+'_target_'+str(target)+"_summary_plot_all_columns.png",dpi=150, bbox_inches='tight')
    plt.close()
    shap.summary_plot(explainer.shap_values(X_test_i.iloc[:,:]),
                  features = X_test_i.iloc[:,:],
                  feature_names=all_columns,
                 plot_type='bar',
                 color='dodgerblue',show=False)

    plt.savefig("image/plot_importance_columns/"+str(year_test)+'_'+str(week_test)+'_t_'+str(t)+'_target_'+str(target)+"_summary_plot2_all_columns.png",dpi=150, bbox_inches='tight')
    plt.close()
    shap.summary_plot(explainer.shap_values(X_test_i.iloc[:,:]),
                  features = X_test_i.iloc[:,:],
                  feature_names=all_columns,
                 plot_type='violin',
                 color='tomato',show=False)

    plt.savefig("image/plot_importance_columns/"+str(year_test)+'_'+str(week_test)+'_t_'+str(t)+'_target_'+str(target)+"_summary_plot3_all_columns.png",dpi=150, bbox_inches='tight')
    plt.close()
    
    with plt.style.context("ggplot"):
        fig = plt.figure(figsize=(9,6))
        ax = fig.add_subplot(111)
        plot_importance(reg, ax=ax, height=0.6, importance_type="weight", max_num_features=12,title='Importancia de las columnas', ylabel='Columnas')
        plt.savefig("image/plot_importance_columns/"+str(year_test)+'_'+str(week_test)+'_t_'+str(t)+'_target_'+str(target)+"_importance_plot_xgb.png",dpi=150, bbox_inches='tight')
        plt.close()
        
    xgb.to_graphviz(reg).render("image/plot_importance_columns/"+str(year_test)+'_'+str(week_test)+'_t_'+str(t)+'_target_'+str(target)+"_tree_plot")
    plt.close()
    
    #categorical values 
    shap.summary_plot(shap_values[:,Categorical_columns_index],
                  features = X_test_i.iloc[:,Categorical_columns_index],
                  feature_names=Categorical_columns,show=False)
    plt.savefig("image/plot_importance_columns/"+str(year_test)+'_'+str(week_test)+'_t_'+str(t)+'_target_'+str(target)+"_summary_plot_categorical_columns.png",dpi=150, bbox_inches='tight')
    plt.close()
    shap.summary_plot(shap_values[:,Categorical_columns_index],
                  features = X_test_i.iloc[:,Categorical_columns_index],
                  feature_names=Categorical_columns,
                 plot_type='bar',
                 color='dodgerblue',show=False)

    plt.savefig("image/plot_importance_columns/"+str(year_test)+'_'+str(week_test)+'_t_'+str(t)+'_target_'+str(target)+"_summary_plot2_categorical_columns.png",dpi=150, bbox_inches='tight')
    plt.close()
    shap.summary_plot(shap_values[:,Categorical_columns_index],
                  features = X_test_i.iloc[:,Categorical_columns_index],
                  feature_names=Categorical_columns,
                 plot_type='violin',
                 color='tomato',show=False)

    plt.savefig("image/plot_importance_columns/"+str(year_test)+'_'+str(week_test)+'_t_'+str(t)+'_target_'+str(target)+"_summary_plot3_categorical_columns.png",dpi=150, bbox_inches='tight')
    plt.close()
    #numerical values 
    shap.summary_plot(shap_values[:,Numeric_columns_index],
                  features = X_test_i.iloc[:,Numeric_columns_index],
                  feature_names=Numeric_columns,show=False)
    plt.savefig("image/plot_importance_columns/"+str(year_test)+'_'+str(week_test)+'_t_'+str(t)+'_target_'+str(target)+"_summary_plot_numeric_columns.png",dpi=150, bbox_inches='tight')
    plt.close()
    shap.summary_plot(shap_values[:,Numeric_columns_index],
                  features = X_test_i.iloc[:,Numeric_columns_index],
                  feature_names=Numeric_columns,
                 plot_type='bar',
                 color='dodgerblue',show=False)

    plt.savefig("image/plot_importance_columns/"+str(year_test)+'_'+str(week_test)+'_t_'+str(t)+'_target_'+str(target)+"_summary_plot2_numeric_columns.png",dpi=150, bbox_inches='tight')
    plt.close()
    shap.summary_plot(shap_values[:,Numeric_columns_index],
                  features = X_test_i.iloc[:,Numeric_columns_index],
                  feature_names=Numeric_columns,
                 plot_type='violin',
                 color='tomato',show=False)

    plt.savefig("image/plot_importance_columns/"+str(year_test)+'_'+str(week_test)+'_t_'+str(t)+'_target_'+str(target)+"_summary_plot3_numeric_columns.png",dpi=150, bbox_inches='tight')
    plt.close()
    