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
from transformers import AutoTokenizer

class llama3_sophgo(BaseModel):
    def __init__(self):
        super().__init__('language/generative/llama3')

        self.devices = [0]
        self.model_path = 'model/model_set/bmodel/language/generative/llama3/llama3-8b_int4_1dev_seq512.bmodel'
        self.tokenizer_path = 'model/model_set/pytorch/language/generative/llama3/token_config'
        self.params_flops = self.get_params_flops()

    def encode_tokens(self, input_text):
        self.history.append({"role": "user", "content": input_text})
        return self.tokenizer.apply_chat_template(self.history, tokenize=True, add_generation_prompt=True)
    
    def get_input(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
        self.tokenizer.decode(self.devices)  # Warm up
        self.system_prompt = 'You are Llama3, a helpful AI assistant.'
        self.EOS = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        self.system = {"role": "system", "content": self.system_prompt}
        self.history = [self.system]
        
        self.input = "Q: Name the planets in the solar system? A: "
        self.tokens = self.encode_tokens(self.input)

    def load_model(self, temperature=1.0, top_p=1.0, repeat_penalty=1.0, repeat_last_n=32, 
                   max_new_tokens=1024, generation_mode="greedy", prompt_mode="prompted", decode_mode="basic"):
        if decode_mode == "basic":
            import model.model_set.models.language.generative.llama3.utils.chat as chat
            # import utils.chat as chat
            self.model = chat.Llama3()
            self.model.init(self.devices, self.model_path)
            self.model.temperature = temperature
            self.model.top_p = top_p
            self.model.repeat_penalty = repeat_penalty
            self.model.repeat_last_n = repeat_last_n
            self.model.max_new_tokens = max_new_tokens
            self.model.generation_mode = generation_mode
            self.model.prompt_mode = prompt_mode
        else:
            raise ValueError(f"decode mode: {decode_mode} is illegal!")

        self.SEQLEN = self.model.SEQLEN

    def get_params_flops(self) -> list:
        'float [params, flops]'

        return [803, float('nan')]

    def inference(self):
        tok_num = 0
        self.answer_cur = ""
        self.answer_token = []
        token = self.model.forward_first(self.tokens) # First token

        full_word_tokens = [] # Following tokens
        while token not in self.EOS and self.model.token_length < self.SEQLEN:
            full_word_tokens.append(token)
            word = self.tokenizer.decode(full_word_tokens, skip_special_tokens=True)
            if "�" in word:
                token = self.model.forward_next()
                tok_num += 1
                continue
            self.answer_token.append(token)
            # print(word, flush=True, end="")
            token = self.model.forward_next()
            tok_num += 1
            full_word_tokens = []

        answer_cur = self.tokenizer.decode(self.answer_token) # Decode final answer
        return tok_num
    