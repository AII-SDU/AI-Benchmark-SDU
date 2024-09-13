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
from diffusers import PNDMScheduler
from model.model_set.models.multimodality.generative.stablediffusionv1_5.utils.stable_diffusion import StableDiffusionPipeline

class stablediffusionv1_5_sophgo(BaseModel):
    def __init__(self):
        super().__init__('multimodality/generative/stablediffusionv1_5')

        self.stage = "singlize"
        self.img_size = (512, 512)
        self.model_path = "model/model_set/bmodel/multimodality/generative/stablediffusionv1_5"
        self.tokenizer = "model/model_set/pytorch/multimodality/generative/stablediffusionv1_5/tokenizer_path"

    def get_input(self):
        self.prompt = "a photo of an astronaut riding a horse on mars"

        self.scheduler = PNDMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                skip_prk_steps=True,
            )

    def load_model(self):
        self.pipeline = StableDiffusionPipeline(
                scheduler = self.scheduler,
                model_path = self.model_path,
                stage = self.stage,
                tokenizer = self.tokenizer,
                dev_id = 0,
                controlnet_name = None,
                processor_name = None,
            ) 

    def get_params_flops(self) -> list:
        'float [params, flops]'

        with open('config.json', 'r') as file:
            config = json.load(file)
            model_info = config.get('model_info', {}).get(self.model_identifier, {})
            params = model_info.get('Params(M)', 'Not available')
            flops = model_info.get('FLOPs(G)', 'Not available')
        return [params, flops]

    def inference(self):
        image = self.pipeline(prompt = self.prompt,
        height = self.img_size[0],
        width = self.img_size[1],
        negative_prompt = "worst quality",
        init_image = None,
        controlnet_img = None,
        strength = 0.7,
        num_inference_steps = 50,
        guidance_scale = 7.5)
        return image
    

    