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

import time
import datetime
import torch

def run_inference(model, iterations, minimum_time, model_performance, event_list, barrier):
    """
    Executes multiple iterations of inference for a single model.

    The `run_inference` function performs inference for a given model, running it for a specified number of iterations 
    or until a minimum amount of time has elapsed. It initializes the model, performs warm-up inference, and then 
    runs the actual inference while monitoring the performance.
    """
    model_infer_stop_event = event_list[0]
    data_copy_completed_event = event_list[1]
    monitor_stop_event = event_list[2]
    
    # model.get_input()
    # model.get_params_flops()
    # model.load_model()
    # torch.musa.init()
    model.get_input()
    model.load_model()
    params, flops = model.get_params_flops()
    
    # Pre-run inference to warm up the device
    for _ in range(int(iterations / 10)):
        model.inference()      

    # Within the inference loop, wait for all processes to reach the barrier, 
    # ensuring they start each iteration of inference simultaneously。
    barrier.wait()
    print(f" - {datetime.datetime.now().strftime('%H:%M:%S.%f')} - Inference for {model.model_identifier} starts now.")

    t_start = time.time()
    iter_ = 0
    elapsed_time = 0
    with torch.no_grad():
        while iter_ < iterations or elapsed_time < minimum_time:
            num = model.inference()
            iter_ += num if 'llama3' in model.model_identifier else 1   # The time efficiency of llama3 is measured by the number of tokens generated per unit of time.
            elapsed_time = time.time() - t_start

    model_performance.extend([model.model_identifier, elapsed_time, iter_, params, flops])

    model_infer_stop_event.set()
    print(f" - {datetime.datetime.now().strftime('%H:%M:%S.%f')} - Inference for {model.model_identifier} ended.")

    monitor_stop_event.wait()

    # Clear events for the next model inference
    monitor_stop_event.clear()
    data_copy_completed_event.clear()
    model_infer_stop_event.clear()     