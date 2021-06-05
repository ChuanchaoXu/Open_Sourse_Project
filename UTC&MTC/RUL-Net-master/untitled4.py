# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 20:42:34 2020

@author: zxj
"""





from  data_processing import data_augmentation, get_PHM08Data


training_data, testing_data, phm_testing_data = get_PHM08Data(save = True)

data_augmentation(files="phm", low=[10, 40, 90, 170], high=[35, 85, 160, 250], plot=True, combine=False)

