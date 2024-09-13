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
from threading import BrokenBarrierError

class DeviceManager:
    """
    Manages the device monitoring process during model inference.

    The DeviceManager is responsible for monitoring the performance metrics of the device 
    (such as utilization, memory usage, and power consumption) during the inference of each model.

    Before the monitoring process begins, it clears the previous monitoring result data and synchronizes 
    with the inference and data handling processes. The device performance is recorded at regular intervals 
    and passed on for data handling once each model's inference is complete.
    """
    def __init__(self, opt, deviceUsage_list, event_list, barrier, devicefctory):

        self.opt = opt
        self.deviceUsage_list = deviceUsage_list

        self.model_infer_stop_event = event_list[0]
        self.data_copy_completed_event = event_list[1]
        self.monitor_stop_event = event_list[2]
        self.barrier = barrier

        self.devicefctory = devicefctory
        self.device = devicefctory.get_device()

    def __clear_monitor_result_data(self):
        self.deviceUsage_list[:] = [['device Utilization', 'device MemoryUsage', 'device Power']]

    def run_monitor(self):
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
        try:
            while True:
                self.__clear_monitor_result_data()       # To avoid empty headers, set them within the loop body first.

                self.barrier.wait()
                print(f" - {datetime.datetime.now().strftime('%H:%M:%S.%f')} - Monitoring starts now.")

                while not self.model_infer_stop_event.is_set():
                    t_start = time.time()
                    device_perf_info = self.device.get_device_perf_info()
                    self.deviceUsage_list.append(device_perf_info)
                    t_elapsed = time.time() - t_start
                    time_to_sleep = max(0, self.opt.device_monitor_interval - t_elapsed)
                    if time_to_sleep == 0:
                        print(f'Warning: The query time for {self.device.device_name} exceeded the device monitoring interval. It is recommended to reduce the value of --device_monitor_interval')
                    time.sleep(time_to_sleep)
                print(f" - {datetime.datetime.now().strftime('%H:%M:%S.%f')} - Monitor has paused.")

                self.data_copy_completed_event.wait()
                self.monitor_stop_event.set()       # The data_copy_completed_event will be passed from here to run_inference.

        except BrokenBarrierError:
            print(f" - {datetime.datetime.now().strftime('%H:%M:%S.%f')} - Monitor has safely exited.")
        


            



            
            
            














    






