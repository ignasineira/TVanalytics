# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 00:09:16 2021

@author: ignasi
"""
# Load modules.
import os
import sys


"""

Parameters:
    o start_year : year since the information starts to be used.
        Type: int
    o start_month : month since the information starts to be used.
        Type: int
This function call: 
    o The best fit parameter for each tuple (segment ,week) ('save_model/tunning_parameters/').
    o X data: calls the function generate_data ('input_model_final_input.csv').

Takes the last week of the database 'input_input_model_final.csv' as the X_test. 
Return:
    o Save a model with for each tuple (target,week) in 'save_model/last_model/'. 
    o Save the predictions with metadata for each tuple (target,week) in 'save_model/last_model/'. 
 **This function is supposed to be trained every week**.
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
    TV.pred_for_last_week_data(start_year=2015,start_month=1)