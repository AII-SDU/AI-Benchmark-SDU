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

import subprocess
from device.device_set.device_base import Device
from enums import DeviceType

class sophgoTPU(Device):
    def __init__(self):
        super().__init__()

        import sophon.sail as sail
        self.sail = sail
        self.dev_id = 0

        self.devicetype=DeviceType.DEVICE_TYPE_SophgoTPU
        device_info = self.get_device_info()
        self.device_name = device_info[0]
        self.device_memory = device_info[1]


    def get_device_info(self) -> list:

        try:
            with open('/proc/bmsophon/card0/chipid', 'r') as file:
                device_name = 'Sophgo'+file.read().strip()
        except ImportError:
            raise ImportError("Not file: /proc/bmsophon/card0/chipid")
        
        device_memory = self.sail.get_dev_stat(0)[0]

        return [device_name, device_memory]

    def get_device_perf_info(self) -> list:

        result = subprocess.run(['cat', '/proc/bmsophon/card0/board_power'], stdout=subprocess.PIPE)
        output = result.stdout.decode('utf-8').strip()
        power_draw = float(output.split()[0])
        utilization = self.sail.get_tpu_util(self.dev_id)
        memory_usage = self.sail.get_dev_stat(self.dev_id)[1]

        return [utilization, memory_usage, power_draw]

    @staticmethod
    def get_device_command():
        return 'bm-smi --start_dev=0 --last_dev=0 --text_format '