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

from process.data_handle_tools import PostTools
import copy
import datetime
from threading import BrokenBarrierError

class DataHandler:
    """
    After each model inference is completed, perform data processing during the inference, including:
    - Power consumption calculation
    - Time efficiency calculation
    - Evaluation score calculation
    - Result plotting

    The process is compatible with both single model test data and multiple model test data.
    """
    def __init__(self, opt, model_performance, deviceUsage_list, event_list, barrier,  devicefctory):

        self.opt = opt
        self.model_performance = model_performance
        self.deviceUsage_list = deviceUsage_list

        self.model_infer_stop_event = event_list[0]
        self.data_copy_completed_event = event_list[1]
        self.monitor_stop_event = event_list[2]
        self.barrier = barrier
        
        # Device information used for result saving.
        self.devicefctory = devicefctory
        self.device = devicefctory.get_device()
        self.device_type = self.device.devicetype
        self.device_name = self.device.device_name
        self.device_memory = self.device.device_memory      

    def run_handler(self):

        matrix_allmodel = []
        deviceUsage_list_all = []
        try:
            while True:
                self.barrier.wait()
                self.model_infer_stop_event.wait() 
                print(f" - {datetime.datetime.now().strftime('%H:%M:%S.%f')} - Data copy starts now.")

                deviceUsage_list_ = copy.deepcopy(self.deviceUsage_list)
                model_performance_ = copy.deepcopy(self.model_performance)
                self.data_copy_completed_event.set()
                print(f" - {datetime.datetime.now().strftime('%H:%M:%S.%f')} - Data copy is complete; processing starts now.")

                model_identifier = model_performance_[0]
                model_performance_ = model_performance_[1:]
                matrix_singlemodel = PostTools.calc_matrix(deviceUsage_list_, model_performance_, self.opt)
                PostTools.post_process(deviceUsage_list_, matrix_singlemodel, self.opt, self.device_type, self.device_name, self.device_memory, post_process_flag = 'single', model_path = model_identifier)

                deviceUsage_list_all.append([model_identifier, deviceUsage_list_])
                matrix_allmodel.append([model_identifier, *matrix_singlemodel])

        except BrokenBarrierError:
            PostTools.post_process(deviceUsage_list_all, matrix_allmodel, self.opt, self.device_type, self.device_name, self.device_memory)
            print(f" - {datetime.datetime.now().strftime('%H:%M:%S.%f')} - DataHandler has safely exited.")



















        
