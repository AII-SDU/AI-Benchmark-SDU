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

from abc import ABC, abstractmethod

class BaseModel:
    """
    All model classes that are compatible must inherit from this base class, 
    and the initialization should implement the properties model_identifier.

    New model implementations should be organized and placed in the `model/model_set/models` directory, 
    categorized according to their paths. Dependency files should be placed in either the `model/model_set/bmodel` 
    directory or the `model/model_set/pytorch` directory, depending on their respective paths.
    """
    def __init__(self, model_identifier):
        """
        self.model_identifier format: 'categories/applications/modelname'
        Example: 'multimodality/generative/stablediffusionv1_5'
        """
        self.model_identifier = model_identifier


    @abstractmethod
    def get_input(self):

        pass

    @abstractmethod
    def load_model(self):

        pass

    @abstractmethod
    def get_params_flops(self) -> list:
        'float [params, flops]'

        pass

    @abstractmethod
    def inference(self):

        pass

