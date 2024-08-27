import os
import json
import time
import argparse
import threading
import multiprocessing

from tools.init_tools import check_device_type
from tools.init_tools import update_config
from tools.init_tools import choose_model

from tools.post_tools import calc_matrix
from tools.post_tools import post_process

from device.device_monitor import DeviceMonitor
from inference.model_inference import ModelInference


def monitor_process_start(opt, device_type, deviceUsage_list, start_event, stop_event):
    monitor = DeviceMonitor(opt=opt, device_type=device_type)  
    monitor.run_monitor(deviceUsage_list, start_event, stop_event)

def process_process_start(model_path, opt, device_type, model_performance, start_event, stop_event):
    inference = ModelInference(model_path=model_path, opt=opt, device_type=device_type)
    inference.run_inference(model_performance, start_event, stop_event)


def main_fun(model_path, opt, DEVICE_TYPE):

    print(f'Running tests using---- {model_path}')

    # deviceUsage_list = [['device Utilization','device MemoryUsage','device Power']] # Store time-series data of device usage
    # model_performance = [] # Store model inference time data

    # inference = ModelInference(model_path=model_path, opt=opt, device_type=DEVICE_TYPE, model_performance=model_performance)
    # monitor = DeviceMonitor( opt=opt, device_type=DEVICE_TYPE, deviceUsage_list=deviceUsage_list)

    # start_event = threading.Event()
    # stop_event = threading.Event()
    # # x_event = threading.Event()

    # monitor_thread = threading.Thread(target=monitor.run_monitor, args=(start_event, stop_event))
    # monitor_thread.daemon = True
    # monitor_thread.start()  # Start the device monitoring thread

    # inference_thread = threading.Thread(target=inference.run_inference, args=(start_event, stop_event))
    # inference_thread.start()  # Start the inference thread

    # # time.sleep(opt.preheating_time+1)
    # time.sleep(10)
    # start_event.set()  
    # inference_thread.join() 
    # monitor_thread.join()
    
    # return deviceUsage_list, model_performance

    # inference = ModelInference(model_path=model_path, opt=opt, device_type=DEVICE_TYPE)
    # monitor = DeviceMonitor( opt=opt, device_type=DEVICE_TYPE)


    with multiprocessing.Manager() as manager:

        deviceUsage_list = manager.list([['device Utilization', 'device MemoryUsage', 'device Power']])  
        model_performance = manager.list()  # Manage the shared list using ‘Manager’

        start_event = multiprocessing.Event()
        stop_event = multiprocessing.Event()

        monitor_process = multiprocessing.Process(target=monitor_process_start, args = (opt, DEVICE_TYPE, deviceUsage_list, start_event, stop_event))
        monitor_process.start()  # Start the device monitoring process

        inference_process = multiprocessing.Process(target=process_process_start, args=(model_path, opt, DEVICE_TYPE, model_performance, start_event, stop_event))

        inference_process.start()  # Start the inference process

        time.sleep(10)
        start_event.set() 

        inference_process.join()
        monitor_process.join()

        return list(deviceUsage_list), list(model_performance)











DEVICE_TYPE = check_device_type()  ##device_type

parser = argparse.ArgumentParser()
parser.add_argument("--testmode", type=int, default=0, help="Test mode. 0 for testing the entire model set; 1 for testing a single model.")
parser.add_argument("--iterations", type=int, default=5000, help="Number of inference iterations.")
parser.add_argument("--iterations_l", type=int, default=50, help="Number of inference iterations,for [stablediffusionv1_5, ]")
parser.add_argument("--minimum_time", type=int, default=180, help="Minimum time for model iterations, in seconds.")
# parser.add_argument("--preheating_time", type=int, default=120, help="Device preheating time, in seconds.")
parser.add_argument('--device_monitor_interval', type=float, default=0.3, help="Device monitoring frequency, in seconds per check.")
parser.add_argument('--bmodel_precision', type=int, default=32, help="Specify the bmodel precision for testing SophgoTPU. Options: 32 or 16.")

opt = parser.parse_args()



if __name__ == "__main__":

    multiprocessing.set_start_method('spawn')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.json')
    base_path = os.path.join(script_dir, 'model')
    
    update_config(config_path, base_path, DEVICE_TYPE) # update config
    with open('config.json', 'r') as file:
        config = json.load(file)
    model_list = config['model_list']

    if opt.testmode == 0:

        matrix_allmodel = []
        deviceUsage_list_allmodel = []
        for model_path in model_list:
            deviceUsage_list, model_performance = main_fun(model_path, opt, DEVICE_TYPE)
            matrix_singlemodel = calc_matrix(deviceUsage_list, model_performance, opt)

            post_process(deviceUsage_list, matrix_singlemodel, opt, DEVICE_TYPE, post_process_flag = 'single', model_path = model_path) # Process the test data for each individual model separately
            print(f'Completed device testing for model {model_path}')

            deviceUsage_list_allmodel.append([model_path, deviceUsage_list])
            matrix_allmodel.append([model_path, *matrix_singlemodel])

        post_process(deviceUsage_list_allmodel, matrix_allmodel, opt, DEVICE_TYPE)  # Process the test data for all models collectively
        print(f'Completed device testing for all models in the Model Benchmark Set')

    if opt.testmode == 1:

        model_path = choose_model(model_list)

        deviceUsage_list, model_performance = main_fun(model_path, opt, DEVICE_TYPE)
        matrix_singlemodel = calc_matrix(deviceUsage_list, model_performance, opt)
        
        post_process(deviceUsage_list, matrix_singlemodel, opt, DEVICE_TYPE, post_process_flag = 'single', model_path = model_path)
        
        print(f'Completed device testing for model---- {model_path}')



