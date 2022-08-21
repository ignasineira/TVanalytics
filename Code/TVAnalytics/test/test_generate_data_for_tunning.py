# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 15:08:56 2021

@author: ignasi
"""

# Load modules.
import os
import sys


"""
This file calls the function generate_data from data_processing
o This test is only for the  tuning process.
The necessary parameters are:
    tunning: True if the data generation is for the tunning process
        Type: Boolean 
    save_metadata: True if I save the database in the input folder
                    False if calls the database saved in the input folder
        Type: Boolean 
        
Within the data generation process, the file 'input\columns_name_final.csv' is
imported, which contains the name of the columns. 
In addition to returning a Dataframe, if tunning==True and save_metadata==True, 
two csv are saved in the folder "output/All_tunning_parameter/". 
The first one contains temporary information for all the rows of 
the dataframe ("X_data_test") and the second file contains all 
the metadata of Canal13 with categorical columns ("X_data_test_categorical_columns").

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
    
    
    df_final=TV.generate_data(save_metadata=True,tunning=True)
    df_final.to_csv('input/df_final.csv',index = False)
    
    