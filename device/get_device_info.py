import subprocess
import pynvml
import subprocess
import re


def get_device_info(DEVICE_TYPE):
    if DEVICE_TYPE == 'NVIDIA':
        pynvml.nvmlInit() 
        device = pynvml.nvmlDeviceGetHandleByIndex(0)  
        device_name = pynvml.nvmlDeviceGetName(device)
        device_memory = pynvml.nvmlDeviceGetMemoryInfo(device).total / (1024 ** 2)  # MB
        pynvml.nvmlShutdown()

    elif DEVICE_TYPE == 'AMD':

        # import pyopencl as cl
        # result = subprocess.check_output(['rocm-smi', '-a'], encoding='utf-8')
        # # print(result)
        # device_name_pattern = re.compile(r'GPU\[\d+\]\s*:\s*Device Name\s*:\s*(.+)')
        # match = device_name_pattern.search(result)
        # device_name = match.group(1).strip()   
        
        # platforms = cl.get_platforms()  
        # platform = platforms[0] 
        # devices = platform.get_devices(cl.device_type.ALL)
        # device_memory = devices[0].global_mem_size / (1024 ** 2)  # MB

        import sys
        sys.path.append("/opt/rocm/libexec/rocm_smi/")
        try:
            import rocm_smi
        except ImportError:
            raise ImportError("Could not import /opt/rocm/libexec/rocm_smi/rocm_smi.py")
        
        rocm_smi.initializeRsmi()
        Devices = rocm_smi.listDevices()

        (memory_usage, memTotal) = rocm_smi.getMemInfo(Devices[0], "vram")
        device_memory = float(memTotal)/1024/1024
        device_name = 'ADM' + rocm_smi.getDeviceName(Devices[0])

    elif DEVICE_TYPE == 'SophgoTPU':
        dev_id = 0
        import sophon.sail as sail

        with open('/proc/bmsophon/card0/chipid', 'r') as file:
            device_name = 'Sophgo'+file.read().strip()
        device_memory = sail.get_dev_stat(dev_id)[0]

    else:
        device_name, device_memory = 'notNVIDIAorAMDorSophgoTPU', 0000

    return device_name, device_memory