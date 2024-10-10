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
import torch_musa
from model.model_set.model_base import BaseModel
import torch
from transformers import BertTokenizer, ErnieModel
from thop import profile

class ernie3_mthreads(BaseModel):
    def __init__(self):
        super().__init__('language/nlp/ernie3')
        self.device = torch.device('musa' if torch.musa.is_available() else 'cpu') 
        self.tokenizer_path = "model/model_set/pytorch/language/nlp/ernie3/vocab"
        self.model_path = "model/model_set/pytorch/language/nlp/ernie3/vocab"
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path)
        
    def get_input(self):
        self.text = "Hello, how are you?"
        self.max_length = 256
        # Tokenize input text
        self.inputs = self.tokenizer(self.text, return_tensors='pt', padding='max_length', 
                                     truncation=True, max_length=self.max_length).to(self.device)
        
    def load_model(self):

        self.model = ErnieModel.from_pretrained(self.model_path).to(self.device)

    def get_params_flops(self) -> list:
        flops, _ = profile(self.model, (self.inputs.input_ids, self.inputs.attention_mask), verbose=False)
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return flops / 1e9 * 2,  params / 1e6
    
    def inference(self):
        with torch.no_grad():  
            outputs = self.model(**self.inputs)
        return outputs
     
    