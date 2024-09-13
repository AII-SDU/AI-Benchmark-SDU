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

from model.model_set.model_base import BaseModel
from llama_cpp import Llama


class llama3_nvidia_amd(BaseModel):
    def __init__(self):
        super().__init__('language/generative/llama3')

    def get_input(self):
        self.input = "Q: Name the planets in the solar system? A: "

    def load_model(self):
        self.llm = Llama(
            model_path="model/model_set/pytorch/language/generative/llama3/ggml-meta-llama-3-8b-Q4_K_M.gguf",
            n_gpu_layers=99,
            #   n_gpu_layers=-1, # Uncomment to use GPU acceleration
            chat_format="llama-3",
            seed=1337, # Uncomment to set a specific seed
            n_ctx=2048, # Uncomment to increase the context window
            verbose=False
            )

    def get_params_flops(self) -> list:

        return [803, float('nan')]


    def inference(self):
        output = self.llm (
                prompt = self.input, # Prompt
                max_tokens=512, # Generate up to 32 tokens, set to None to generate up to the end of the context window
                stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
                echo=True # Echo the prompt back in the output
            )
        completion_tokens = output['usage']['completion_tokens']
        return completion_tokens
    
    