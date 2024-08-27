import json
import os
import subprocess

def check_device_type():
    device_commands = {
        'NVIDIA': ['nvidia-smi'],
        'AMD': ['rocm-smi'],
        'SophgoTPU': ['bm-smi', '--start_dev=0', '--last_dev=0', '--text_format']
    }

    for device_type, command in device_commands.items():
        try:
            subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return device_type
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    return 'No NVIDIA, AMD or SophgoTPU'

def generate_model_list(base_dir, DEVICE_TYPE):
    if DEVICE_TYPE == 'SophgoTPU':
        base_dir = os.path.join(base_dir, 'bmodel')
    elif DEVICE_TYPE in ['NVIDIA', 'AMD']:
        base_dir = os.path.join(base_dir, 'pytorch')

    dir_list = []

    for category in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category)
        if os.path.isdir(category_path):
            for application in os.listdir(category_path):
                application_path = os.path.join(category_path, application)
                if os.path.isdir(application_path):
                    for model in os.listdir(application_path):
                        model_path = os.path.join(application_path, model)
                        if os.path.isdir(model_path):
                            dir_list.append(f"{category}/{application}/{model}")

    return sorted(dir_list)

def choose_model(model_list):
    print("Please select a model:")
    for i, model in enumerate(model_list, start=1):
        print(f"{i}. {model}")

    choice = input("Enter the model number: ")
    while not choice.isdigit() or int(choice) < 1 or int(choice) > len(model_list):
        print("Invalid option, please enter again.")
        choice = input("Enter the model number: ")
    
    model_path = model_list[int(choice) - 1]
    return model_path

def update_config(config_path, base_path, DEVICE_TYPE):
    model_list = generate_model_list(base_path, DEVICE_TYPE)

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}

    existing_model_list = config.get('model_list', [])
    existing_model_info = config.get('model_info', {})

    updated_model_list = sorted(set(existing_model_list + model_list))

    updated_model_info_dict = {model: existing_model_info.get(model, {
        "Params(M)": None,
        "FLOPs(G)": None,
        "Latency_stand": None,
        "Energy_stand": None
    }) for model in updated_model_list}

    config['model_list'] = updated_model_list
    config['model_info'] = updated_model_info_dict

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    return
