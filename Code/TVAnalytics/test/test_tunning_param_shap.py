# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 18:46:26 2021

@author: ignasi
"""

# Load modules.
import os
import sys
import multiprocessing


"""
The pipeline_tunning_param_shap function of tunning_param_shap calls from 
the input folder calls the following files: "df_final.csv" and "columns_predict".
And from the save_model folder: 
"tunning_parameters / best_parameter_t_X_target_Y.json '


Parameters:
    
    * n_week_train: number of weeks to perform the tuning parameters 
            Type: int
    * lower_bound : lower bound of the columns that you enter in the 
    models depending on the importance
            Type: int

                    
Output: 
    plot importance: 
        "image\Shap_tunning\week_X\plot_summary_importance_columns_shap_target_Y_t_X_summary_plot2.png"
    summary_importance:
        "output\Shap_tunning\week_X\summary_importance_columns_shap_target_X_t_Y_lower_bound_W.csv"
    columns_shap:
        'output/Shap_tunning/columns_shap.json'
        
    Prediccition: 
        "output\Shap_tunning\week_X\model_predict_xgb_tunning_target_Y_pre_0_year...csv"


"""

if __name__ == "__main__":
    # Load local package.
    module_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    
    if module_path not in sys.path:
        sys.path.append(module_path)
    
    import TVAnalytics as TV
    # Set your local data directory.
    path = 'C:/Users/ignasi/Desktop/TVAnalytics'
    
    os.chdir(path)
    lower_bound=0.00005;
    n_week_train=15
    
    TV.pipeline_tunning_param_shap(lower_bound=lower_bound,
                                   n_week_train=n_week_train)