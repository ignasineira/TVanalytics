# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 23:52:08 2021

@author: ignasi
"""

import os

def generate_new_directory(path, include_week=True):
    if include_week:
        for i in range(1,7):
            try:
                os.makedirs(path+'\week_'+str(i))
            except OSError:
                print()
    else:
        try:
            os.makedirs(path)
        except OSError:
            print()