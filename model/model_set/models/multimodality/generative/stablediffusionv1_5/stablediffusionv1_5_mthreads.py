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
import torch
import json
from model.model_set.model_base import BaseModel
from diffusers import StableDiffusionPipeline

class stablediffusionv1_5_mthreads(BaseModel):
    def __init__(self):
        super().__init__('multimodality/generative/stablediffusionv1_5')

        self.device = torch.device('musa' if torch.musa.is_available() else 'cpu')
        self.model_path = "model/model_set/pytorch/multimodality/generative/stablediffusionv1_5/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9"

    def get_input(self):
        self.prompt = "a photo of an astronaut riding a horse on mars"

    def load_model(self):
        self.pipeline = StableDiffusionPipeline.from_pretrained(self.model_path, torch_dtype=torch.float32).to(self.device)

    def get_params_flops(self) -> list:
        'float [params, flops]'
        with open('config.json', 'r') as file:
            config = json.load(file)
            model_info = config.get('model_info', {}).get(self.model_identifier, {})
            params = model_info.get('Params(M)', 'Not available')
            flops = model_info.get('FLOPs(G)', 'Not available')
        return [params, flops]

    def inference(self):
        image = self.pipeline(prompt = self.prompt).images[0]
        return image
    
    
    