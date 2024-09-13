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
from model.model_set.model_base import BaseModel
from diffusers import StableDiffusionPipeline

class stablediffusionv1_5_nvidia_amd(BaseModel):
    def __init__(self):
        super().__init__('multimodality/generative/stablediffusionv1_5')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = "model/model_set/pytorch/multimodality/generative/stablediffusionv1_5/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9"

    def get_input(self):
        self.prompt = "a photo of an astronaut riding a horse on mars"

    def load_model(self):
        self.pipeline = StableDiffusionPipeline.from_pretrained(self.model_path, torch_dtype=torch.float32).to(self.device)

    def get_params_flops(self) -> list:
        from fvcore.nn import FlopCountAnalysis
        dummy_input_unet = torch.randn(1, 4, 64, 64).to(self.device)  # Corresponding to the input shape of UNet
        dummy_input_vae = torch.randn(1, 3, 64, 64).to(self.device)   # Corresponding to the input shape of VAE
        dummy_input_text = torch.randint(0, 77, (1, 77)).to(self.device)
        dummy_timestep = torch.tensor([1.0], device=self.device)  # A hypothetical time step
        dummy_text = torch.randint(0, 77, (1, 77)).to(self.device)
        encoder_hidden_states = self.pipeline.text_encoder(dummy_text)[0]
        flops_unet = FlopCountAnalysis(self.pipeline.unet, (dummy_input_unet, dummy_timestep, encoder_hidden_states))
        # Calculate FLOPs for VAE
        flops_vae = FlopCountAnalysis(self.pipeline.vae, dummy_input_vae)
        # FLOPs for calculating text encoders
        flops_text_encoder = FlopCountAnalysis(self.pipeline.text_encoder, dummy_input_text)
        # Calculate total FLOPs
        total_flops = flops_unet.total() + flops_vae.total() + flops_text_encoder.total()

        # Calculate the parameter count of UNet
        unet_params = sum(p.numel() for p in self.pipeline.unet.parameters())
        # Calculate the parameter quantity of VAE
        vae_params = sum(p.numel() for p in self.pipeline.vae.parameters())
        # Calculate the parameter count of CLIP text encoder
        text_encoder_params = sum(p.numel() for p in self.pipeline.text_encoder.parameters())
        # Calculate the total number of parameters
        total_params = unet_params + vae_params + text_encoder_params
        return [total_flops / 1e9 * 2, total_params / 1e6]

    def inference(self):
        image = self.pipeline(prompt = self.prompt).images[0]
        return image

    

    

    