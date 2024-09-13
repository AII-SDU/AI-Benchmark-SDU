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

import sys

from device.device_set.device_base import Device
from enums import DeviceType


class AMD(Device):
    def __init__(self):
        super().__init__()

        sys.path.append("/opt/rocm/libexec/self.rocm_smi/")
        try:
            import rocm_smi 
        except ImportError:
            raise ImportError("Could not import /opt/rocm/libexec/rocm_smi/rocm_smi.py")

        rocm_smi.initializeRsmi()
        self.rocm_smi = rocm_smi
        self.devices = rocm_smi.listdevices()

        self.devicetype=DeviceType.DEVICE_TYPE_AMD
        device_info = self.get_device_info()
        self.device_name = device_info[0]
        self.device_memory = device_info[1]

    def get_device_info(self) -> list:
        (memory_usage, memTotal) = self.rocm_smi.getMemInfo(self.devices[0], "vram")
        device_memory = float(memTotal) / 1024 / 1024  # MB
        device_name = 'Sophgo'+ self.rocm_smi.getDeviceName(self.devices[0])

        return [device_name, device_memory]

    def get_device_perf_info(self) -> list:
        utilization = float(self.rocm_smi.getGpuUse(self.devices[0]))
        (memory_usage, memTotal) = self.rocm_smi.getMemInfo(self.devices[0], "vram")
        memory_usage = float(memory_usage) / 1024 / 1024  # MB
        power_draw = float(self.rocm_smi.getPower(self.devices[0])['power'])

        return [utilization, memory_usage, power_draw]

    @staticmethod
    def get_device_command():
        return 'rocm-smi'