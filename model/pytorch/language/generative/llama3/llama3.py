import torch
import time
from transformers import AutoTokenizer

class llama3:
    def __init__(self, 
                mode='tpu', 
                model_path='model/bmodel/language/generative/llama3/llama3-8b_int4_1dev_seq512.bmodel',
                tokenizer_path='model/pytorch/language/generative/llama3/token_config',
                devid='0',
                temperature=1.0,
                top_p=1.0,
                repeat_penalty=1.0,
                repeat_last_n=32,
                max_new_tokens=1024,
                generation_mode="greedy",
                prompt_mode="prompted",
                decode_mode="basic",
                enable_history=False ):

        self.mode = mode
        # Device initialization
        if mode == 'tpu':
            self.devices = [int(d) for d in devid.split(",")]
            self.tokenizer_path = tokenizer_path
            self.model_path = model_path

            # Load tokenizer
            print(f"Loading tokenizer from {self.tokenizer_path} ...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
            self.tokenizer.decode([0])  # Warm up

            # System settings
            self.system_prompt = 'You are Llama3, a helpful AI assistant.'
            self.EOS = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            self.system = {"role": "system", "content": self.system_prompt}
            self.history = [self.system]
            self.enable_history = enable_history

            # Load model
            self.load_model(temperature, top_p, repeat_penalty, repeat_last_n, 
                            max_new_tokens, generation_mode, prompt_mode, decode_mode)
        if mode == 'gpu':
            from llama_cpp import Llama
            self.llm = Llama(
            model_path="model/pytorch/language/generative/llama3/ggml-meta-llama-3-8b-Q4_K_M.gguf",
            n_gpu_layers=99,
            #   n_gpu_layers=-1, # Uncomment to use GPU acceleration
            chat_format="llama-3",
            seed=1337, # Uncomment to set a specific seed
            n_ctx=2048, # Uncomment to increase the context window
            verbose=False
            )
        # else:
        #     raise ValueError("Mode should be either 'gpu' or 'tpu'")

    def load_model(self, temperature, top_p, repeat_penalty, repeat_last_n, 
                   max_new_tokens, generation_mode, prompt_mode, decode_mode):
        if decode_mode == "basic":
            import model.pytorch.language.generative.llama3.utils.chat as chat
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

    def encode_tokens(self, input_text):
        self.history.append({"role": "user", "content": input_text})
        return self.tokenizer.apply_chat_template(self.history, tokenize=True, add_generation_prompt=True)

    # def forward(self):
    #     if self.mode == 'gpu':
    #         from llama_cpp import Llama
    #         llm = Llama(
    #         model_path="./models/ggml-meta-llama-3-8b-Q4_K_M.gguf",
    #         n_gpu_layers=99,
    #         #   n_gpu_layers=-1, # Uncomment to use GPU acceleration
    #         chat_format="llama-3",
    #         seed=1337, # Uncomment to set a specific seed
    #         n_ctx=2048, # Uncomment to increase the context window
    #         )
    #         start_time = time.time()
    #         output = llm(
    #             "Q: Name the planets in the solar system? A: ", # Prompt
    #             max_tokens=512, # Generate up to 32 tokens, set to None to generate up to the end of the context window
    #             stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
    #             echo=True # Echo the prompt back in the output
    #         ) # Generate a completion, can also call create_completion
    #         end_time = time.time()
    #         inference_time = end_time - start_time
    #         completion_tokens = output['usage']['completion_tokens']
    #         return inference_time, completion_tokens
        
    #     elif self.mode == 'tpu':
    #         tok_num = 0
    #         self.answer_cur = ""
    #         self.answer_token = []
    #         first_start = time.time()

    #         # First token
    #         token = self.model.forward_first(self.tokens)
    #         first_end = time.time()

    #         full_word_tokens = []
    #         # Following tokens
    #         while token not in self.EOS and self.model.token_length < self.SEQLEN:
    #             full_word_tokens.append(token)
    #             word = self.tokenizer.decode(full_word_tokens, skip_special_tokens=True)
    #             if "�" in word:
    #                 token = self.model.forward_next()
    #                 tok_num += 1
    #                 continue

    #             self.answer_token.append(token)
    #             print(word, flush=True, end="")
                
    #             token = self.model.forward_next()
    #             tok_num += 1
    #             full_word_tokens = []

    #         # Counting time
    #         next_end = time.time()
    #         first_duration = first_end - first_start
    #         next_duration = next_end - first_end

    #         # Decode final answer
    #         answer_cur = self.tokenizer.decode(self.answer_token)
    #         return answer_cur, first_duration, next_duration, tok_num
    #     else:
    #         raise ValueError("Mode should be either 'gpu' or 'tpu'")
    
    def forward(self, input_text):
        if self.mode == 'gpu':

            output = self.llm (
                prompt = input_text, # Prompt
                max_tokens=512, # Generate up to 32 tokens, set to None to generate up to the end of the context window
                stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
                echo=True # Echo the prompt back in the output
            ) # Generate a completion, can also call create_completion

            completion_tokens = output['usage']['completion_tokens']
            return  completion_tokens
        
        elif self.mode == 'tpu':
            tokens = self.encode_tokens(input_text)
            tok_num = 0
            self.answer_cur = ""
            self.answer_token = []
            # First token
            token = self.model.forward_first(tokens)

            full_word_tokens = []
            # Following tokens
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

            # Decode final answer
            answer_cur = self.tokenizer.decode(self.answer_token)
            
            return tok_num
        else:
            raise ValueError("Mode should be either 'gpu' or 'tpu'")

    def get_response(self, input_text):
        self.tokens = self.encode_tokens(input_text)
        return self.forward()
    
    def get_response_thread(self, input_text, start_event, stop_event):
        tokens = self.encode_tokens(input_text)
        return self.forward_thread(tokens, start_event, stop_event)

if __name__ == "__main__":
    mode = 'tpu' 
    model = llama3(mode=mode)
    if mode == 'gpu':
        # t_start = time.time()
        # iterations = 128
        # for _ in range(iterations):
        #     with torch.no_grad():
        #         outputs = model.forward()
        # elapsed_time = time.time() - t_start
        # latency = elapsed_time / iterations * 1000
        # FPS = 1000 / latency
        # print(f"FPS: {FPS:.2f}")
        # flops, params = model.count_parameters_and_flops()
        # print(f"FLOPs: {flops} GFLOPs")
        # print(f"Parameters: {params} Million")
        outputs = model.forward_thread()
        print(f"Tokens per second: {outputs[1] / outputs[0]:.2f} token/s")
    
    elif mode == 'tpu':

        question = "Name the planets in the solar system?"
        response, first_duration, next_duration, tok_num = model.forward(question)
        
        # print(f"Answer: {response}")
        # print(f"First token latency: {first_duration:.2f} s")
        # print(f"Total latency: {next_duration:.2f} s")
        print(f"Total tokens generated: {tok_num}")
        print(f"Tokens per second: {tok_num / next_duration:.2f} token/s")


