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

from model.model_factory import ModelFactory
from model.model_inference import run_inference

import time
import multiprocessing

class ModelManager:
    """
    Manages the iterative inference process for all models.

    The ModelManager is responsible for performing iterative inference on a set of models.

    Before starting the inference of each model, it clears previous inference result data, 
    sets the appropriate number of iterations for inference, and creates and manages the model inference processes.
    """
    def __init__(self, opt, model_path_list, model_performance, event_list, barrier,  devicefctory):
        self.opt = opt
        self.model_performance = model_performance

        self.barrier = barrier
        self.event_list = event_list

        self.devicefctory = devicefctory
        self.device = devicefctory.get_device()
        self.device_type = self.device.devicetype

        modelfactory = ModelFactory(self.device_type)
        self.model_list = modelfactory.get_model_list(model_path_list)

    def __clear_inference_result_data(self):
        self.model_performance[:] = []

    def __get_iterations(self, model_identifier):
        # If the model is a generative model, perform fewer iterations for inference.
        if 'generative' not in model_identifier:
            return self.opt.many_iterations
        else:
            return self.opt.few_iterations

    def run_models_inference(self):
        """
        - The `barrier.wait()` ensures that the model inference, device monitoring, and data handling processes start 
        simultaneously at the beginning of each model's inference iteration, keeping the processes synchronized.

        - The inference process controls the main workflow. After completing the inference for each model, 
        it triggers `model_infer_stop_event` to notify the data handling and monitoring processes to proceed.

        - Data handling begins after the inference ends, copying both inference and monitoring data, and then triggers 
        `data_copy_completed_event` to notify the monitoring process, achieving synchronization between the processes.

        - The monitoring process, upon receiving both the inference and data copy completion signals, triggers 
        `monitor_stop_event` to inform the inference process that both data copy and monitoring have been completed, 
        allowing it to proceed with the next model.

        - After the final model’s inference, monitoring, and data handling are completed, the system waits briefly to 
        ensure that both the monitor and data handler processes reach `barrier.wait()`. Following that, the barrier 
        is aborted, allowing both processes to exit safely through the `BrokenBarrierError` exception.

        - The use of `monitor_stop_event` ensures that the inference process requires minimal event setting to know 
        that both data copying and monitoring are complete.

        - The `monitor_stop_event` also ensures that the monitoring process stops before `model_infer_stop_event` is 
        cleared, preventing the `while not self.model_infer_stop_event.is_set()` loop from prematurely exiting, 
        avoiding any issues with the monitoring loop getting stuck.
        """
        for model in self.model_list:
            self.__clear_inference_result_data()
            iterations = self.__get_iterations(model.model_identifier)
            minimum_time = self.opt.minimum_time

            model_process = multiprocessing.Process(target=run_inference, args=(model, iterations, minimum_time, self.model_performance, self.event_list, self.barrier))
            model_process.start()
            model_process.join()

        #Ensure that both the monitor process and the data handler process reach the barrier.wait() after the inference of the last model is complete.
        time.sleep(10)    
        # Aborts the barrier, causing any waiting processes (monitor and data handler) to exit when all model inferences are complete.
        self.barrier.abort()    
