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

from device.device_set.pymtml_SDU.pymtml import pymtml

from device.device_set.device_base import Device
from enums import DeviceType


class MTHREADS(Device):
    def __init__(self):
        super().__init__()

        pymtml.mtmlInit() 
        self.device_handle = pymtml.mtmlDeviceGetHandleByIndex(0)

        self.devicetype=DeviceType.DEVICE_TYPE_MTHREADS
        device_info = self.get_device_info()
        self.device_name = device_info[0]
        self.device_memory = device_info[1]

    def get_device_info(self) -> list:
        device_name = pymtml.mtmlDeviceGetName(self.device_handle)
        device_memory = pymtml.mtmlDeviceGetMemoryInfo(self.device_handle).total / 1024 / 1024  # MB

        return [device_name, device_memory]

    def get_device_perf_info(self) -> list:
        memory_usage = pymtml.mtmlDeviceGetMemoryInfo(self.device_handle).used / 1024 / 1024  # MB
        utilization = pymtml.mtmlDeviceGetUtilizationRates(self.device_handle).gpu
        power_draw = pymtml.mtmlDeviceGetPowerUsage(self.device_handle) / 1000.0  # W

        return [utilization, memory_usage, power_draw]
    
    @staticmethod
    def get_device_command():
        return 'mthreads-gmi'
    


    