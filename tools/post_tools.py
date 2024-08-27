import json
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import pynvml

from device.get_device_info import get_device_info


def calc_matrix(deviceUsage_list, model_performance, opt):
    time, iter, params, flops = model_performance[0], model_performance[1], model_performance[2], model_performance[3]

    # Total energy consumption
    column4 = [row[2] for row in deviceUsage_list[1:]]
    time_interval_s = opt.device_monitor_interval
    total_energy = sum(power * time_interval_s for power in column4) / 1000  # KW*s = KJ

    # Average energy consumption per iteration
    PerIter_energy = total_energy / iter * 1000  # Ws = J

    # Average power and memory usage
    memory_usage = [row[1] for row in deviceUsage_list[1:]]
    power_usage = [row[2] for row in deviceUsage_list[1:]]
    average_memory_usage = sum(memory_usage) / len(memory_usage)
    average_power_usage = sum(power_usage) / len(power_usage)

    # Latency and FPS
    latency = time / iter * 1000
    FPS = 1000 / latency

    matrix_singlemodel = [params, flops, FPS, latency, total_energy, PerIter_energy, average_memory_usage, average_power_usage]

    return matrix_singlemodel





def format_matrix(matrix):
    formatted_matrix = []
    for row in matrix:
        formatted_row = [f"{item:.2f}" if isinstance(item, (int, float)) else str(item) for item in row]
        formatted_matrix.append(formatted_row)
    return formatted_matrix

def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def save_benchmark_results(df, device_name, savedir, filename_suffix):
    df.to_csv(os.path.join(savedir, f'benchmark_{device_name}{filename_suffix}.csv'), index=False)
    plt.savefig(os.path.join(savedir, f'benchmark_{device_name}{filename_suffix}.png'))

def benchmark_score(matrix, DEVICE_TYPE, device_name, weight=True):

    matrix = pd.DataFrame(data=matrix, columns=['Model', 'Params(M)', 'FLOPs(G)', 'FPS', 'Latency(ms)', 'Energy(KJ)', 'PerIterEnergy(J)', 'AverageMemoryUsage(M)', 'AveragePowerUsage(W)'])
    
    config_path = 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    if DEVICE_TYPE == 'NVIDIA' and '2080' in device_name:

        save_df = matrix[['Model', 'Params(M)', 'FLOPs(G)', 'Latency(ms)', 'PerIterEnergy(J)']]
        model_info = config.get('model_info', {})
        for _, row in save_df.iterrows():
            model = row['Model']
            if model in model_info:
                if model_info[model]['Params(M)'] is None:
                    model_info[model]['Params(M)'] = row['Params(M)']
                if model_info[model]['FLOPs(G)'] is None:
                    model_info[model]['FLOPs(G)'] = row['FLOPs(G)']
                if model_info[model]['Latency_stand'] is None:
                    model_info[model]['Latency_stand'] = row['Latency(ms)']
                if model_info[model]['Energy_stand'] is None:
                    model_info[model]['Energy_stand'] = row['PerIterEnergy(J)']
        config['model_info'] = model_info
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

    with open(config_path, 'r') as f:
        config = json.load(f)
    model_info = config.get('model_info', {})
    data = [{'Model': model, 'Latency_stand': info.get('Latency_stand'), 'Energy_stand': info.get('Energy_stand')} for model, info in model_info.items()]
    standard = pd.DataFrame(data, columns=['Model', 'Latency_stand', 'Energy_stand'])

    energy_score = []
    latency_score = []
    basic_score = 1000

    monitor_data_b = pd.DataFrame(columns=['Model', 'Latency_stand', 'Energy_stand'])
    monitor_data_b['Model'] = matrix.loc[:, 'Model']
    monitor_data_b['Latency_stand'] = matrix.loc[:, 'Latency(ms)']
    monitor_data_b['Energy_stand'] = matrix.loc[:, 'PerIterEnergy(J)']

    monitor_data_b_array = monitor_data_b.values
    standard_array = standard.values
    
    for i in range(len(monitor_data_b_array)):
        values = standard_array[standard_array[:, 0] == monitor_data_b_array[i, 0]][0]
        latency_score.append( values[1] / monitor_data_b_array[i, 1] * basic_score)
        energy_score.append( values[2] / monitor_data_b_array[i, 2] * basic_score)

    latency_score = np.array(latency_score)
    energy_score = np.array(energy_score)

    if weight:
        device_latency_score = latency_score.mean()
        device_energy_score = energy_score.mean()
    else:
        device_latency_score = sum(weight[monitor_data_b_array[j, 0]] * latency_score[j] for j in range(len(latency_score)))
        device_energy_score = sum(weight[monitor_data_b_array[j, 0]] * energy_score[j] for j in range(len(energy_score)))

    return device_latency_score, device_energy_score, energy_score, latency_score

def plot_matrix(matrix_allmodel, deviceUsage_list_allmodel, opt, device_latency_score, device_energy_score, energy_score, latency_score, post_process_flag, device_name, device_memory):
    device_name = device_name.replace('/', '_')
    savedir = f'output/savefiles_iter{opt.iterations}/'
    ensure_dir_exists(savedir)

    device_info = f"{device_name}; DEVICE Memory: {device_memory:.2f} MB; DEVICE_latency_score: {device_latency_score:.2f}; DEVICE_energy_score: {device_energy_score:.2f}"

    matrix_allmodel = np.array(matrix_allmodel, dtype=object)
    energy_score = np.array(energy_score, dtype=float).reshape(-1, 1)
    latency_score = np.array(latency_score, dtype=float).reshape(-1, 1)
    combined_matrix = np.hstack((matrix_allmodel, latency_score, energy_score))
    combined_matrix = format_matrix(combined_matrix)

    header = ['Model', 'Params (M)', 'FLOPs (G)', 'FPS', 'Latency (ms)', 'Energy (KJ)', 'PerIterEnergy (J)', 'AvgMemUsage (M)', 'AvgPowerUsage (W)', 'latency_score', 'energy_score']
    combined_matrix = [header] + combined_matrix

    df = pd.DataFrame(combined_matrix[1:], columns=combined_matrix[0])
    df.set_index('Model', inplace=True)

    num_plots = len(deviceUsage_list_allmodel)
    fig = plt.figure(figsize=(20, 4 * (num_plots + 2)))
    spec = gridspec.GridSpec(nrows=num_plots + 2, ncols=4, height_ratios=[1] + [2] * (num_plots + 1), width_ratios=[1, 4, 4, 4])

    ax_info = fig.add_subplot(spec[0, :])
    ax_info.text(0.5, 0.5, device_info, horizontalalignment='center', verticalalignment='center', fontsize=20)
    ax_info.axis('off')

    ax_table = fig.add_subplot(spec[1, :])
    table = ax_table.table(cellText=combined_matrix, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1)
    ax_table.axis('off')

    for key, cell in table.get_celld().items():
        if key[1] == 0:
            cell.set_text_props(ha='right')
            cell.set_width(0.15)

    for i, (title, data) in enumerate(deviceUsage_list_allmodel):
        title = os.path.basename(title)
        data = np.array(data[1:])
        time = np.arange(len(data))

        ax_info = fig.add_subplot(spec[i + 2, 0])
        ax_info.text(0.5, 0.5, title, horizontalalignment='left', verticalalignment='center', fontsize=16)
        ax_info.axis('off')

        ax1 = fig.add_subplot(spec[i + 2, 1])
        ax1.plot(time, data[:, 0], color='b')
        ax1.set_xlabel(f'Time ({opt.device_monitor_interval}s)')
        ax1.set_ylabel('GPU Utilization (%)')
        # ax1.set_ylim(bottom=0)

        ax2 = fig.add_subplot(spec[i + 2, 2])
        ax2.plot(time, data[:, 1], color='r')
        ax2.set_xlabel(f'Time ({opt.device_monitor_interval}s)')
        ax2.set_ylabel('Memory Usage (MB)')
        # ax2.set_ylim(bottom=0)

        ax3 = fig.add_subplot(spec[i + 2, 3])
        ax3.plot(time, data[:, 2], color='g')
        ax3.set_xlabel(f'Time ({opt.device_monitor_interval}s)')
        ax3.set_ylabel('Power Usage (W)')
        # ax3.set_ylim(bottom=0)

    filename_suffix = '' if post_process_flag == 'all' else os.path.basename(deviceUsage_list_allmodel[0][0])
    save_benchmark_results(df, device_name, savedir, filename_suffix)

    return

def post_process(deviceUsage_list, matrix, opt, DEVICE_TYPE, post_process_flag = 'all', model_path = None):

    if post_process_flag == 'single':
        deviceUsage_list = [[model_path, deviceUsage_list]]
        matrix = [[model_path,*matrix]]

    device_name, device_memory = get_device_info(DEVICE_TYPE)

    device_latency_score, device_energy_score, energy_score, latency_score  = benchmark_score(matrix, DEVICE_TYPE, device_name)
    plot_matrix(matrix, deviceUsage_list, opt, device_latency_score, device_energy_score, energy_score, latency_score, post_process_flag, device_name, device_memory) 
