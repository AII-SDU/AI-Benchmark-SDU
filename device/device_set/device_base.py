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

from abc import ABC, abstractmethod

class Device(ABC):
    """
    All device classes that are compatible must inherit from this base class, 
    and the initialization should implement the properties devicetype, device_name, and device_memory.
    """
    def __init__(self):
        """
        slef.devicetype=DeviceType.DEVICE_TYPE_XXXX
        device_info = self.get_device_info()
        self.device_name = device_info[0]
        self.device_memory = device_info[1]
        """
        pass

    def get_device_name(self):  
        return self.device_name

    @abstractmethod
    def get_device_info(DEVICE_TYPE) -> list:  
        """    
        return [device_name, device_memory]
        """

        pass

    @abstractmethod
    def get_device_perf_info(self) -> list:
        """
        return [utilization, memory_usage, power_draw]
        """

        pass

    @abstractmethod
    def get_device_command(self):
        """
        return 'xx-smi'
        """
        pass


