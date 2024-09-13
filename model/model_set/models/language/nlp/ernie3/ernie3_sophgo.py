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
from model.model_set.model_base import BaseModel
from transformers import BertTokenizer, ErnieModel
import sophon.sail as sail


class ernie3_sophgo(BaseModel):
    def __init__(self):
        super().__init__('language/nlp/ernie3')

        self.devices = 0
        self.model_path = 'model/model_set/bmodel/language/nlp/ernie3/ernie3_1684x_f32.bmodel'     
        self.tokenizer_path = "model/model_set/pytorch/language/nlp/ernie3/vocab"
        
    def get_input(self):
        self.text = "Hello, how are you?"
        self.max_length = 256
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path)
        self.inputs = self.tokenizer(self.text, return_tensors='pt', padding='max_length', 
                                     truncation=True, max_length=self.max_length)
        self.input_ids = self.inputs['input_ids'].to(dtype=torch.float32).numpy()
       
    def load_model(self):
        self.model = sail.Engine(self.model_path, self.devices, sail.IOMode.SYSIO)
        self.graph_name = self.model.get_graph_names()[0]
        input_name_img  = self.model.get_input_names(self.graph_name)
        self.input_data_dict  = {input_name_img [0]: self.input_ids }

    def get_params_flops(self) -> list:
        'float [params, flops]'

        with open('config.json', 'r') as file:
            config = json.load(file)
            model_info = config.get('model_info', {}).get(self.model_identifier, {})
            params = model_info.get('Params(M)', 'Not available')
            flops = model_info.get('FLOPs(G)', 'Not available')
        return [params, flops]

    def inference(self):
        output = self.model.process(self.graph_name, self.input_data_dict)
        return output

    

    