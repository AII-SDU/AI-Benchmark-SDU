'''
Copyright (c) 2024, 山东大学智能创新研究院(Academy of Intelligent Innovation)

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
'''
# Copyright (c) Academy of Intelligent Innovation.
# License-Identifier: BSD 2-Clause License
# AI Benchmark SDU Team

import os
import importlib
from enums import DeviceType
import datetime

class ModelFactory:
    """
    Load all models used for testing.
    """
    def __init__(self, DEVICE_TYPE):
        self.device_type = DEVICE_TYPE

    def __get_nvi_amd_model_list(self, model_path_list):
        models = []
        for model_path in model_path_list:
            module_name = os.path.basename(model_path) + '_nvidia_amd'

            module_path = f"model.model_set.models.{model_path.replace('/', '.')}.{module_name.lower()}"
            module = importlib.import_module(module_path)
            model_class = getattr(module, module_name)

            model = model_class() 
            models.append(model)

        return models
    
    def __get_sophgo_model_list(self, model_path_list):
        models = []
        for model_path in model_path_list:
            module_name = os.path.basename(model_path) + '_sophgo'

            module_path = f"model.model_set.models.{model_path.replace('/', '.')}.{module_name.lower()}"
            module = importlib.import_module(module_path)
            model_class = getattr(module, module_name)

            model = model_class() 
            models.append(model)

        return models
    
    def __get_mthreads_model_list(self, model_path_list):
        models = []
        if 'language/generative/llama3' in model_path_list:
            model_path_list.remove('language/generative/llama3')
            print(f" - {datetime.datetime.now().strftime('%H:%M:%S.%f')} - The model *language/generative/llama3* is currently not deployed on {self.device_type}.")
        if 'multimodality/classification/clip' in model_path_list:
            model_path_list.remove('multimodality/classification/clip')
            print(f" - {datetime.datetime.now().strftime('%H:%M:%S.%f')} - The model *multimodality/classification/clip* is currently not deployed on {self.device_type}.")
            
        for model_path in model_path_list:
            module_name = os.path.basename(model_path) + '_mthreads'

            module_path = f"model.model_set.models.{model_path.replace('/', '.')}.{module_name.lower()}"
            module = importlib.import_module(module_path)
            model_class = getattr(module, module_name)

            model = model_class() 
            models.append(model)

        return models
    
    def get_model_list(self, model_path_list):

        if self.device_type in [DeviceType.DEVICE_TYPE_NVIDIA , DeviceType.DEVICE_TYPE_AMD]:
            return self.__get_nvi_amd_model_list(model_path_list)

        elif self.device_type == DeviceType.DEVICE_TYPE_SophgoTPU:
            return self.__get_sophgo_model_list(model_path_list)
        
        elif self.device_type == DeviceType.DEVICE_TYPE_MTHREADS:
            return self.__get_mthreads_model_list(model_path_list)














