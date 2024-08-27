
import os
import time
import torch
import requests
from transformers import BertTokenizer, ErnieModel
# from tpu_perf.infer import SGInfer
from thop import profile
import numpy as np

def download_model_weights(model_path):
    if not os.path.exists(os.path.join(model_path, 'pytorch_model.bin')):
        print(f"The weight file does not exist, downloading weights from Hugging Face")
        model_url = "https://huggingface.co/nghuyong/ernie-3.0-medium-zh/resolve/main/pytorch_model.bin?download=true"
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(os.path.join(model_path, 'pytorch_model.bin'), 'wb') as f:
                f.write(response.content)
            print("Weight download completed.")
        else:
            print("Weight download failed, please check network connection or URL.")

class ernie3:
    def __init__(self, mode='gpu', text="Hello, how are you?", max_length=256, model_path='model/pytorch/language/nlp/ernie3/vocab', tokenizer_path='model/pytorch/language/nlp/ernie3/vocab'):
        self.mode = mode
        self.text = text
        self.max_length = max_length
        self.tokenizer_path = tokenizer_path
        self.model_path = model_path
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        
        if mode == 'gpu':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            download_model_weights(model_path)
            self.model = ErnieModel.from_pretrained(model_path).to(self.device)
            self.inputs = self.tokenizer(text=self.text, return_tensors='pt', padding='max_length', max_length=self.max_length).to(self.device)
        elif mode == 'tpu':
            self.inputs = self.tokenizer(text=text, return_tensors='pt', padding='max_length', max_length=max_length)
            self.input_ids = self.inputs['input_ids'].numpy().astype(np.int32)
    
        else:
            raise ValueError("Mode should be either 'gpu' or 'tpu'")


    def count_parameters_and_flops(self):
        flops, _ = profile(self.model, (self.inputs.input_ids, self.inputs.attention_mask), verbose=False)
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return flops / 1e9 * 2,  params / 1e6


    def forward(self):
        if self.mode == 'gpu':
            outputs = self.model(**self.inputs)
            return outputs
        elif self.mode == 'tpu':
            return self.input_ids
        else:
            raise ValueError("Mode should be either 'gpu' or 'tpu'")



