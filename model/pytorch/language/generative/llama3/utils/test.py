
import time
from transformers import AutoTokenizer

class Llama3():
    def __init__(self, 
                 model_path, 
                 tokenizer_path="../support/token_config", 
                 devid='0', 
                 temperature=1.0, 
                 top_p=1.0, 
                 repeat_penalty=1.0, 
                 repeat_last_n=32, 
                 max_new_tokens=1024, 
                 generation_mode="greedy", 
                 prompt_mode="prompted", 
                 decode_mode="basic", 
                 enable_history=True):
        
        # devid
        self.devices = [int(d) for d in devid.split(",")]
        self.tokenizer_path = tokenizer_path
        self.model_path = model_path

        # load tokenizer
        print(f"Load {self.tokenizer_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path, trust_remote_code=True
        )

        # warm up
        self.tokenizer.decode([0])

        # preprocess parameters, such as prompt & tokenizer
        self.system_prompt = 'You are Llama3, a helpful AI assistant.'
        self.EOS = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        self.system = {"role": "system", "content": self.system_prompt}
        self.history = [self.system]
        self.enable_history = enable_history

        # load model
        self.load_model(temperature, top_p, repeat_penalty, repeat_last_n, 
                        max_new_tokens, generation_mode, prompt_mode, decode_mode)

    def load_model(self, temperature, top_p, repeat_penalty, repeat_last_n, 
                max_new_tokens, generation_mode, prompt_mode, decode_mode):
            if decode_mode == "basic":
                import chat
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

    def encode_tokens(self, input_text):
        self.history.append({"role":"user","content":input_text})
        return self.tokenizer.apply_chat_template(self.history, tokenize=True, add_generation_prompt=True)

    def forward(self, tokens):
        tok_num = 0
        self.answer_cur = ""
        self.answer_token = []

        # First token
        first_start = time.time()
        token = self.model.forward_first(tokens)
        first_end = time.time()

        full_word_tokens = []
        # Following tokens
        while token not in self.EOS and self.model.token_length < self.SEQLEN:
            full_word_tokens.append(token)
            word = self.tokenizer.decode(full_word_tokens, skip_special_tokens=True)
            if "�" in word:
                token = self.model.forward_next()
                tok_num += 1
                continue

            self.answer_token += [token]
            print(word, flush=True, end="")
            token = self.model.forward_next()
            tok_num += 1
            full_word_tokens = []

        # counting time
        next_end = time.time()
        first_duration = first_end - first_start
        next_duration = next_end - first_end

        # Decode final answer
        answer_cur = self.tokenizer.decode(self.answer_token)
        return answer_cur, first_duration, next_duration, tok_num

    def get_response(self, input_text):
        tokens = self.encode_tokens(input_text)
        return self.forward(tokens)


