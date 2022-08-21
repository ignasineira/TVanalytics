# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 23:33:13 2021

@author: ignasi
"""

# Load modules.
import os
import sys


"""
This function calls all the predictions for the tunning process, selec the
some models and plot the MAPE changing the way it is grouped and
 save the images in 'image/All_tunning_parameter/'.

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
    
    
    TV.main_plot_tunning()