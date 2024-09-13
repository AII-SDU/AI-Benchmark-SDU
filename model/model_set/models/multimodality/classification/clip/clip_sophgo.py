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

import torch
import json
import numpy as np
import sophon.sail as sail
from model.model_set.model_base import BaseModel
from model.model_set.models.multimodality.classification.clip.utils.simpletokenizer import tokenize_tpu

class clip_sophgo(BaseModel):
    def __init__(self):
        super().__init__('multimodality/classification/clip')

        self.text = ["a diagram", "a dog", "a cat"]
        self.input_shape =(1, 3, 224, 224)
        self.text_net_batch_size = 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_model_path = 'model/model_set/bmodel/multimodality/classification/clip/clip_image_vitb32_bm1684x_f16.bmodel'
        self.text_model_path = 'model/model_set/bmodel/multimodality/classification/clip/clip_text_vitb32_bm1684x_f16.bmodel'

    def get_input(self):
        self.image_input = np.random.randn(*self.input_shape).astype(np.float32)
        self.text_input = self.encode_text(tokenize_tpu(self.text))

    def load_model(self):
        self.image_net = sail.Engine(self.image_model_path, 0, sail.IOMode.SYSIO)
        self.text_net = sail.Engine(self.text_model_path, 0, sail.IOMode.SYSIO)
        self.graph_name_img = self.image_net.get_graph_names()[0]
        input_name_img  = self.image_net.get_input_names(self.graph_name_img)
        self.input_data_dict_img  = {input_name_img [0]: self.image_input }
        self.graph_name_text = self.text_net.get_graph_names()[0]
        input_name_text  = self.text_net.get_input_names(self.graph_name_text)
        self.input_data_dict_text  = {input_name_text [0]: self.text_input }

    def encode_text(self, text):
        text_batch = text.shape[0]
        if text_batch > self.text_net_batch_size:
            for start_idx in range(0, text_batch, self.text_net_batch_size):
                end_idx = min(start_idx + self.text_net_batch_size, text_batch)  # Ensure end_idx does not exceed text_batch
                batch_slice = text[start_idx:end_idx]
                if batch_slice.shape[0] < self.text_net_batch_size:
                    padding_size = self.text_net_batch_size - batch_slice.shape[0]
                    batch_slice = np.concatenate([batch_slice, np.zeros((padding_size, *batch_slice.shape[1:]), dtype=batch_slice.dtype)], axis=0)
            return batch_slice
        else:
            return text
        
    def get_params_flops(self) -> list:
        'float [params, flops]'

        with open('config.json', 'r') as file:
            config = json.load(file)
            model_info = config.get('model_info', {}).get(self.model_identifier, {})
            params = model_info.get('Params(M)', 'Not available')
            flops = model_info.get('FLOPs(G)', 'Not available')
        return [params, flops]

    def inference(self):
        img_results = self.image_net.process(self.graph_name_img, self.input_data_dict_img)
        txt_results = self.text_net.process(self.graph_name_text , self.input_data_dict_text)
        return img_results, txt_results
    
    
    