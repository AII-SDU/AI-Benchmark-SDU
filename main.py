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

import os
import argparse
import datetime
import multiprocessing
from multiprocessing import Barrier, Process

from process.init_tools import InitTools
from device.device_factory import DeviceFactory
from device.device_manager import DeviceManager
from model.model_manager import ModelManager
from process.data_handler import DataHandler

def inference_process_test(opt, model_path_list, model_performance, event_list, barrier, devicefctory):
    model_manager = ModelManager(opt, model_path_list, model_performance, event_list, barrier, devicefctory)
    model_manager.run_models_inference()

def monitor_process_start(opt, deviceUsage_list, event_list, barrier, devicefctory):
    device_manager = DeviceManager(opt, deviceUsage_list, event_list, barrier, devicefctory)
    device_manager.run_monitor()

def handler_process_start(opt, model_performance, deviceUsage_list, event_list, barrier, devicefctory):
    data_handler = DataHandler(opt, model_performance, deviceUsage_list, event_list, barrier, devicefctory)
    data_handler.run_handler()

parser = argparse.ArgumentParser()
parser.add_argument("--testmode", type=int, default=0, help="Test mode. 0 for testing the entire model set; 1 for testing a single model.")
parser.add_argument("--many_iterations", type=int, default=5000, help="Inference Count for Non-Generative Models.")
parser.add_argument("--few_iterations", type=int, default=20, help="Inference Count for Generative Models.")
parser.add_argument("--minimum_time", type=int, default=150, help="Minimum time for model iterations, in seconds.")
# parser.add_argument("--preheating_time", type=int, default=120, help="Device preheating time, in seconds.")
parser.add_argument('--device_monitor_interval', type=float, default=0.3, help="Device monitoring frequency, in seconds per check.")
parser.add_argument('--bmodel_precision', type=int, default=32, help="Specify the bmodel precision for testing SophgoTPU. Options: 32 or 16.")

opt = parser.parse_args()

if __name__ == "__main__":

    # Initial preparation of the program.
    devicefctory = DeviceFactory()
    multiprocessing.set_start_method('spawn')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config = InitTools.load_config_and_update(script_dir) 
    model_path_list = config['model_list']
    if opt.testmode == 1:
        model_path_list = [InitTools.choose_model(model_path_list)]     

    with multiprocessing.Manager() as manager:

        barrier = Barrier(3) 
        event_list = [multiprocessing.Event() for _ in range(3)]
        deviceUsage_list = manager.list()
        model_performance = manager.list()

        # Initialize and start the monitor and handler processes.
        monitor_process = multiprocessing.Process(target=monitor_process_start, args=(opt, deviceUsage_list, event_list, barrier, devicefctory))
        handler_process = multiprocessing.Process(target=handler_process_start, args=(opt, model_performance, deviceUsage_list, event_list, barrier, devicefctory))

        monitor_process.start()
        handler_process.start()

        inference_process_test(opt, model_path_list, model_performance, event_list, barrier,  devicefctory)

        monitor_process.join()
        handler_process.join()

        print(f" - {datetime.datetime.now().strftime('%H:%M:%S.%f')} - All models have been used to complete the {devicefctory.check_device_type().name} testing.")
