import time
import pynvml
import subprocess
import re

class DeviceMonitor:
    def __init__(self, opt, device_type):
        self.opt = opt
        self.device_type = device_type

        if device_type == 'NVIDIA':
            pynvml.nvmlInit() 

        elif device_type == 'AMD':  
            # self.pattern_match = [
            #     re.compile(r'device use \(%\):\s*(\d+\.?\d*)'),
            #     re.compile(r'device Memory Allocated \(VRAM%\):\s*(\d+\.?\d*)'),
            #     re.compile(r'Average Graphics Package Power \(W\):\s*(\d+\.?\d*)')
            # ]

            import sys
            sys.path.append("/opt/rocm/libexec/rocm_smi/")
            try:
                import rocm_smi 
            except ImportError:
                raise ImportError("Could not import /opt/rocm/libexec/rocm_smi/rocm_smi.py")
            
            rocm_smi.initializeRsmi()
            self.rocm_smi = rocm_smi
            self.Devices = rocm_smi.listDevices()


        elif device_type == 'SophgoTPU':
            self.dev_id = 0
            import sophon.sail as sail
            self.sail = sail

    def get_device_perf_info(self):
        if self.device_type == 'NVIDIA':
            return self.get_nvidia_perf_info()
        elif self.device_type == 'AMD':
            return self.get_amd_perf_info()
        elif self.device_type == 'SophgoTPU':
            return self.get_sophgo_perf_info()
        else:
            raise ValueError(f"Unsupported device type: {self.device_type}")

    def get_nvidia_perf_info(self):
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        return [utilization.gpu, mem_info.used / 1024**2, power_draw]

    def get_amd_perf_info(self):
        # result = subprocess.run(['rocm-smi', '-a'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # output = result.stdout
        # utilization = float(self.pattern_match[0].search(output).group(1)) if self.pattern_match[0].search(output) else 0
        # memory_usage = float(self.pattern_match[1].search(output).group(1)) if self.pattern_match[1].search(output) else 0
        # power_draw = float(self.pattern_match[2].search(output).group(1)) if self.pattern_match[2].search(output) else 0
        # return [utilization, memory_usage, power_draw]

        utilization = float(self.rocm_smi.getGpuUse(self.Devices[0]))
        (memory_usage, memTotal) = self.rocm_smi.getMemInfo(self.Devices[0], "vram")
        memory_usage = float(memory_usage)/1024/1024 #MB
        power_draw = float(self.rocm_smi.getPower(self.Devices[0])['power'])

        return [utilization, memory_usage, power_draw]

    def get_sophgo_perf_info(self):
        result = subprocess.run(['cat', '/proc/bmsophon/card0/board_power'], stdout=subprocess.PIPE)
        output = result.stdout.decode('utf-8').strip()
        power_draw = float(output.split()[0])
        utilization = self.sail.get_tpu_util(self.dev_id)
        memory_usage = self.sail.get_dev_stat(self.dev_id)[1]
        return [utilization, memory_usage, power_draw]

    def run_monitor(self, deviceUsage_list, start_event, stop_event):
        start_event.wait()
        print('=========Device monitor has started=========')

        while not stop_event.is_set():
            t_start = time.time()
            device_perf_info = self.get_device_perf_info()
            deviceUsage_list.append(device_perf_info)
            t_elapsed = time.time() - t_start
            time_to_sleep = max(0, self.opt.device_monitor_interval - t_elapsed)
            if self.opt.device_monitor_interval < t_elapsed:
                print(f'{self.device_type} query time exceeded device monitoring interval. It is recommended to reduce the value of --device_monitor_interval.')
            time.sleep(time_to_sleep)


if __name__ == "__main__":
        import sys
        sys.path.append("/opt/rocm/libexec/rocm_smi/")
        try:
            import rocm_smi
        except ImportError:
            raise ImportError("Could not import /opt/rocm/libexec/rocm_smi/rocm_smi.py")
        
        rocm_smi.initializeRsmi()
        Devices = rocm_smi.listDevices()

        utilization = rocm_smi.getGpuUse(Devices[0])
        (memory_usage, memTotal) = rocm_smi.getMemInfo(Devices[0], "vram")
        memory_usage = float(memory_usage)/1024/1024 #MB
        memTotal = float(memTotal)/1024/1024
        power_draw = rocm_smi.getPower(Devices[0])['power']
        device_name = rocm_smi.getDeviceName(Devices[0])

        print([utilization, memory_usage, power_draw, memTotal, device_name])




# 
