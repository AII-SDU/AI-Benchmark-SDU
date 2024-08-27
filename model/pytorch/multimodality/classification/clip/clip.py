import os
import requests
import time
import numpy as np
import torch
from PIL import Image
from thop import profile
import cv2
import os
import torch

def download_model_weights(models_path):
    models_path = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(models_path, 'ViT-B-32.pt')):
        print(f"The weight file does not exist, downloading weights from Hugging Face")
        model_url = "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(os.path.join(models_path, 'ViT-B-32.pt'), 'wb') as f:
                f.write(response.content)
            print("Weight download completed.")
        else:
            print("Weight download failed, please check network connection or URL")

class clip:
    def __init__(self, input_shape=(1, 3, 224, 224), mode='gpu', text=["a diagram", "a dog", "a cat"],
                 model_path="model/pytorch/multimodality/classification/clip/ViT-B-32.pt"):
        self.mode = mode
        self.text = text
        self.input_shape = input_shape
        self.img = np.random.randn(*input_shape).astype(np.float32)
        self.text_net_batch_size = 1

        if mode == 'gpu':
            # from model import build_model
            from model.pytorch.multimodality.classification.clip.model import build_model
            from model.pytorch.multimodality.classification.clip.simpletokenizer import SimpleTokenizer as _Tokenizer
            jit = False
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            download_model_weights(model_path)
            model = torch.jit.load(model_path, map_location=self.device if jit else "cpu").eval()
            state_dict = None
            self.model = build_model(state_dict or model.state_dict()).to(self.device)
            # model = CLIP()
            # self.model,_ = load(model_path, device=self.device)
            # model = build_model(state_dict or model.state_dict()).to(self.device)
            # self.texts = clip.tokenize(self.text).to(self.device)
            self.img = torch.tensor(self.img).to(torch.float32).to(self.device) 
            _tokenizer = _Tokenizer()
            sot_token = _tokenizer.encoder["<|startoftext|>"]
            eot_token = _tokenizer.encoder["<|endoftext|>"]
            all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in self.text]
            context_length: int = 77
            truncate = False
            result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

            for i, tokens in enumerate(all_tokens):
                if len(tokens) > context_length:
                    if truncate:
                        tokens = tokens[:context_length]
                        tokens[-1] = eot_token
                    else:
                        raise RuntimeError(f"Input {self.text[i]} is too long for context length {context_length}")
                result[i, :len(tokens)] = torch.tensor(tokens)
                
            self.texts = result.to(self.device)

        elif mode == 'tpu':
            from model.pytorch.multimodality.classification.clip.simpletokenizer import tokenize_tpu
            self.text_input = tokenize_tpu(text)
        else:
            raise ValueError("Mode should be either 'gpu' or 'tpu'")
        
    def count_parameters_and_flops(self):
        flops, _ = profile(self.model, (self.img, self.texts), verbose=False)
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return flops / 1e9 * 2,  params / 1e6
    
    def softmax(self, x, axis=None):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)

    def topk(self, x, k):
        indices = np.argpartition(x, -k)[-k:]
        indices = indices[np.argsort(-x[indices])]
        return x[indices], indices

    def preprocess_cpu(self, image):
        image = cv2.resize(image, (self.image_resolution, self.image_resolution), interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype('float32')
        image = (image/255-self.mean)/self.std
        image = np.transpose(image, (2, 0, 1))
        return image

    def preprocess(self, image):
        start_time = time.time()
        image = self.preprocess_cpu(image)
        self.preprocess_time += time.time() - start_time
        return image
    
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

    def predict(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        image_features /= np.linalg.norm(image_features,axis=-1, keepdims=True)
        text_features /= np.linalg.norm(text_features,axis=-1, keepdims=True)
        similarity = self.softmax((100.0 * np.dot(image_features , text_features.T)),axis=-1) #Calculate similarity and convert it into a probability distribution  
        values, indices = self.topk(similarity[0],min(len(text), self.top_k))
        return values, indices
    
    def forward(self):
        if self.mode == 'gpu':
            image_features = self.model.encode_image(self.img)
            text_features = self.model.encode_text(self.texts) 
            return image_features, text_features
        elif self.mode == 'tpu':
            image_input = self.img
            text_input = self.encode_text(self.text_input)
            return [image_input, text_input]
        else:
            raise ValueError("Mode should be either 'gpu' or 'tpu'")
    

