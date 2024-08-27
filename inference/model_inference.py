import os
import torch
import time
import numpy as np
import importlib.util
import glob
from thop import profile
import json


class ModelInference:
    def __init__(self, model_path, opt, device_type):
        
        self.model_path = model_path
        self.model_name = os.path.basename(self.model_path)
        self.opt = opt
        self.device_type = device_type

        if self.device_type in ['NVIDIA','AMD']:
            self.device = 'cuda'

        # Retrieve input data
        self.input = self._get_input()
        # Load the model
        self.model = self._load_model()
        # Calculate performance parameters
        self.params, self.flops = self._calculate_params_and_flops()


    def _get_input(self):

        vision_model_input_shapes = {
        "yolov5s": (1, 3, 640, 640),
        "ghostnet": (1, 3, 640, 640),
        "unet": (1, 3, 640, 640),
        "resnet": (1, 3, 256, 256),
        "mobilenetv2": (1, 3, 256, 256),
        "bisenetv2": (1, 3, 1024, 2048), 
        "ViT": (1, 3, 224, 224)
        } 

        if self.model_name in ['llama3']:
            with open('inference/questions_list.csv', 'r') as file:
                input = [line.strip() for line in file]

        elif self.device_type == 'SophgoTPU':

            if 'vision' in self.model_path:
                input_shape = vision_model_input_shapes[self.model_name]
                input = np.random.randn(*input_shape).astype(np.float32)               

            else: 
                print(self.model_path)
                model_input_path = os.path.join('model/pytorch', self.model_path,  self.model_name + '.py')
                spec = importlib.util.spec_from_file_location("model_module", model_input_path)
                model_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(model_module)
                model_class = getattr(model_module, self.model_name)
                model_instance = model_class(mode = 'tpu')
                input = model_instance.forward()

        else:
            device = torch.device(self.device)

            if 'vision' in self.model_path:
                input_shape = vision_model_input_shapes[self.model_name]
                input = torch.randn(*input_shape).to(device)

            else:
                input = None

        return input
        

    def _load_model(self):
        self.model_name = os.path.basename(self.model_path)

        if self.device_type == 'SophgoTPU':
            import sophon.sail as sail

            file_pattern = os.path.join('model/bmodel', self.model_path, '*'+str(self.opt.bmodel_precision)+'.bmodel')
            model_script_path = glob.glob(file_pattern)

            if self.model_name == 'clip':
                image_model_path = None
                text_model_path = None
                for model_file in model_script_path:
                    if 'image' in model_file:
                        image_model_path = model_file
                    elif 'text' in model_file:
                        text_model_path = model_file
                image_net = sail.Engine(image_model_path, 0, sail.IOMode.SYSIO)
                text_net = sail.Engine(text_model_path, 0, sail.IOMode.SYSIO)
                model=[image_net, text_net]

            elif self.model_name in ['llama3', 'stablediffusionv1_5']:
                model_script_path = os.path.join('model/pytorch', self.model_path, self.model_name + '.py')
                spec = importlib.util.spec_from_file_location("model_module", model_script_path)
                model_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(model_module)
                model_class = getattr(model_module, self.model_name)
                model = model_class( mode = 'tpu')

            else:
                model = sail.Engine(model_script_path[0], 0, sail.IOMode.SYSIO)

        else:
            model_script_path = os.path.join('model/pytorch', self.model_path, self.model_name + '.py')
            spec = importlib.util.spec_from_file_location("model_module", model_script_path)
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)
            model_class = getattr(model_module, self.model_name)

            if 'language' in self.model_path or 'multimodality' in self.model_path: 
                model = model_class(mode = 'gpu')
            else:
                model = model_class()
                model.eval()
                model.to(self.device)

        return model

    def _calculate_params_and_flops(self):

        if self.device_type == 'NVIDIA':
            if 'vision' in self.model_path:
                flops, params = profile(self.model, inputs=(self.input,), verbose=False)
                flops, params =  flops / 1e9 * 2, params / 1e6
            elif self.model_name == 'llama3':
                flops, params = float('nan'), 803
            elif self.model_name == 'stablediffusionv1_5':
                flops, params = 746.898079232, 1066.235307
            elif 'language' in self.model_path or 'multimodality' in self.model_path:
                flops, params = self.model.count_parameters_and_flops()
        else:
            with open('config.json', 'r') as file:
                config = json.load(file)
            model_info = config.get('model_info', {}).get(self.model_path, {})

            params = model_info.get('Params(M)', 'Not available')
            if 'llama3' in self.model_path:
                flops = float('nan')
            else:
                flops = model_info.get('FLOPs(G)', 'Not available')
        return params, flops

    def model_inference_torch_v(self, model, input, opt, model_performance, start_event, stop_event):
        start_event.wait()
        print('========= Model Inference Started =========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        iter_ = 0

        while iter_ < opt.iterations or elapsed_time < opt.minimum_time:
            model(input)
            iter_ += 1
            elapsed_time = time.time() - t_start

        torch.cuda.synchronize()
        torch.cuda.synchronize()

        model_performance.extend([elapsed_time, iter_])
        time.sleep(0.2)  
        stop_event.set()

    def model_inference_torch_l(self, model, opt, model_performance, start_event, stop_event):
        if self.model_name in ['llama3']:
            with open('inference/questions_list.csv', 'r') as file:
                questions = [line.strip() for line in file]

            start_event.wait()
            print('========= Model Inference Started =========')
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            t_start = time.time()
            iter_ = 0
            for input_data in questions:
                token_counts = model.forward(input_data)
                iter_ += token_counts
            elapsed_time = time.time() - t_start
            torch.cuda.synchronize()
            torch.cuda.synchronize()

        elif self.model_name in ['stablediffusionv1_5']:
            start_event.wait()
            print('========= Model Inference Started =========')
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            t_start = time.time()
            iter_ = 0
            while iter_ < opt.iterations_l or elapsed_time < opt.minimum_time:
                model.forward()
                iter_ += 1
                elapsed_time = time.time() - t_start
            torch.cuda.synchronize()
            torch.cuda.synchronize()

        else:
            start_event.wait()
            print('========= Model Inference Started =========')
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            t_start = time.time()
            iter_ = 0
            while iter_ < opt.iterations or elapsed_time < opt.minimum_time:
                model.forward()
                iter_ += 1
                elapsed_time = time.time() - t_start
            torch.cuda.synchronize()
            torch.cuda.synchronize()

        model_performance.extend([elapsed_time, iter_])
        time.sleep(0.2)  
        stop_event.set()

    def create_input_data_dict(self, input_name, input_list):
        input_data_dict = {}
        
        for name, input_data in zip(input_name, input_list):
            input_array = np.array(input_data)

            if input_array.ndim == 1:
                input_array = input_array.reshape(1, -1)
            elif input_array.ndim == 0:
                input_array = input_array.reshape(1, 1)
            
            input_data_dict[name] = input_array
        
        return input_data_dict

    def model_inference_TPU(self, model, input_data, opt, model_performance, start_event, stop_event):

        if self.model_name in ['llama3']:
            start_event.wait()
            print('========= Model Inference Started =========')
            t_start = time.time()
            iter_ = 0
            for input in input_data:
                print('llama3', input)
                token_counts = model.forward(input)
                iter_ += token_counts
            elapsed_time = time.time() - t_start

        elif self.model_name in ['stablediffusionv1_5']:
            start_event.wait()
            print('========= Model Inference Started =========')
            t_start = time.time()
            iter_ = 0
            elapsed_time = 0
            while iter_ < opt.iterations_l or elapsed_time < opt.minimum_time:
                output = model.forward()
                iter_ += 1
                elapsed_time = time.time() - t_start

        else:
            input_list = [input_data] if not isinstance(input_data, list) else input_data
            model_list = [model] if not isinstance(model, list) else model
            input_data_dicts = []
            graph_names = [] 
            for model, input_data in zip(model_list, input_list):
                # Let's say there is only one graph per model, get the graph name
                graph_name = model.get_graph_names()[0]
                graph_names.append(graph_name)
                # Creates a dictionary of input data for the current model
                input_data_dict = self.create_input_data_dict(model.get_input_names(graph_name), [input_data])
                input_data_dicts.append(input_data_dict)

            start_event.wait()
            print('========= Model Inference Started =========')
            t_start = time.time()
            iter_ = 0
            elapsed_time = 0
            while iter_ < opt.iterations or elapsed_time < opt.minimum_time:
                outputs = []
                for model, input_data_dict, graph_name in zip(model_list, input_data_dicts, graph_names):

                    output = model.process(graph_name, input_data_dict)
                    outputs.append(output)
                
                iter_ += 1
                elapsed_time = time.time() - t_start

        model_performance.extend([elapsed_time, iter_])
        time.sleep(0.2)  
        stop_event.set()

    def run_inference(self, model_performance, start_event, stop_event):
        if self.device_type == 'SophgoTPU':
            self.model_inference_TPU(self.model, self.input, self.opt, model_performance, start_event, stop_event)
        else:
            if 'language' in self.model_path or 'multimodality' in self.model_path:
                self.model_inference_torch_l(self.model, self.opt, model_performance, start_event, stop_event)  
            else:
                self.model_inference_torch_v(self.model, self.input, self.opt, model_performance, start_event, stop_event)

        model_performance.extend([self.params, self.flops])

