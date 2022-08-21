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
The pipeline_tunning_param function of tunning_param calls calls 
the function generate_data from data_processing, 
then iteratively separates the database to train one model per week. 


Parameters:
    
    * n_week_train: number of weeks to perform the tuning parameters 
    * start_year : year since you start using information
            Type: int
    * start_month : month since you start using information
            Type: int
    * generate_new_df : It allows you to save dataframe X, 
                        includes train and test
        Type: True or False;
    * week_list: parameters allow you to choose the future weeks you want to train. 
        Type: List()
    * The target_list parameters allow you to choose the targets to train.
        Type: List()
        
    *param_tuning : dictionary with the hyperparameters to be tested 
                    along with a list of the values ​​to be tested.
                    
                    
                    
For each tuple week of the year, the segment stores the predictions 
in a .csv file in the folder 'output "All_tunning_parameter_week" 
according to the weeks in the future to be predicted. 

This function allows you to parallelize the training.





Examples:
pipeline_tunning_param(param_tuning,save_metadata=True, 
                        n_week_train=15,start_year=2015
                        start_month=1,generate_new_df=True,
                        week_list=[1,2,3,4,5,6], target_list=[0])
pipeline_tunning_param(param_tuning,save_metadata=False, 
                        n_week_train=15,start_year=2015,
                        start_month=1,generate_new_df=False,
                        week_list=[1,2,3,4,5,6], target_list=[1])


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
    save_metadata=True;
    n_week_train=5;
    start_year=2019;start_month=6;
    generate_new_df=False;
    #week_list=[1,2,3,4,5,6]; 
    week_list=[1]; 
    target_list=[0];
    
    param_tuning = {
        'learning_rate': [0.02],
        'max_depth': [10],
        'n_estimators': [100,150],
        'n_jobs': [(multiprocessing.cpu_count() // 2)-3],
        'colsample_bytree': [1],
        'colsample_bylevel':[1],
        'seed': [1]
    }
    
    
    TV.pipeline_tunning_param(param_tuning,
                              save_metadata=save_metadata, 
                              n_week_train=n_week_train,
                              start_year=start_year,
                              start_month=start_month,
                              generate_new_df=generate_new_df,
                              week_list=week_list,
                              target_list=target_list)
    