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

from device.device_set.device_nvidia import NVIDIA
from device.device_set.device_amd import AMD
from device.device_set.device_sophgo import sophgoTPU
from device.device_set.device_mthreads import MTHREADS
from enums import DeviceType

import subprocess

class DeviceFactory:
    def __init__(self):
        pass

    @staticmethod
    def check_device_type():
        device_classes = {
            DeviceType.DEVICE_TYPE_NVIDIA: NVIDIA,
            DeviceType.DEVICE_TYPE_AMD: AMD,
            DeviceType.DEVICE_TYPE_SophgoTPU: sophgoTPU,
            DeviceType.DEVICE_TYPE_MTHREADS: MTHREADS
        }

        for device_type, device_class in device_classes.items():
            device_command = device_class.get_device_command ()     # device_class.get_device_command is staticmethod
            try:
                subprocess.run(device_command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                return device_type
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue

        return DeviceType.DEVICE_TYPE_MAX
    
    @staticmethod
    def get_device(device_type=None):
        if device_type is None:
            device_type = DeviceFactory.check_device_type()

        if device_type == DeviceType.DEVICE_TYPE_NVIDIA:
            return NVIDIA()

        elif device_type == DeviceType.DEVICE_TYPE_AMD:
            return AMD()

        elif device_type == DeviceType.DEVICE_TYPE_SophgoTPU:
            return sophgoTPU()
        
        elif device_type == DeviceType.DEVICE_TYPE_MTHREADS:
            return MTHREADS()

        else:
            return None












