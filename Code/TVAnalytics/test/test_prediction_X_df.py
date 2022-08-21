# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 02:54:55 2021

@author: ignasi
"""

# Load modules.
import os
import sys
import pandas as pd

"""
Parameters:
    o df : X data to predict
    Type: Dataframe
This function imports: 
    o The models saved in the 'save_model/tunning_parameters/' folder. 
    o Call the function generate_data to transform the data.
Returns:
    o Dataframe with the predictions for each tuple (week, target) 
    and metadata as block, day and program code to be able to 
    identify the prediction. 

Comment: 
    this function can be used to generate predictions by changing
    some metadata, but the user must include these changes in the Dataframe. 
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
    df_X=pd.read_csv('input/X_test.csv',header=None)
    
    Y_test = TV.pred_X_test(df_X)