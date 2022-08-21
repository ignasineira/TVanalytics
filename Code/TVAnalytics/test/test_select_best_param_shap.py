# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 23:13:52 2021

@author: ignasi
"""

# Load modules.
import os
import sys


"""
This function calls all the predictions for the tunning process, selec the
model with the minimum WMAPE for the testing weeks for tuple (week,target),
and save the columns in a json file in the folder 
‘input/columns_train_models/columns_train_models.json’. 

Parameters:
    **None**

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
    
    
    TV.main_select_best_params_after_shap()