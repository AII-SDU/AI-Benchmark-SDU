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

import json
import os

class InitTools:
    
    @staticmethod
    def generate_model_list(base_dir):
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
    
    @staticmethod
    def choose_model(model_list):
        print("Please select a model:")
        for i, model in enumerate(model_list, start=1):
            print(f"{i}. {model}")

        choice = input("Enter the model number: ")
        while not choice.isdigit() or not (1 <= int(choice) <= len(model_list)):
            print("Invalid option, please enter again.")
            choice = input("Enter the model number: ")
        
        return model_list[int(choice) - 1]
    
    @staticmethod
    def update_config(config_path, base_path):
        model_list = InitTools.generate_model_list(base_path)

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}

        existing_model_list = config.get('model_list', [])
        existing_model_info = config.get('model_info', {})

        updated_model_list = sorted(set(existing_model_list + model_list))

        updated_model_info = {
            model: existing_model_info.get(model, {
                "Params(M)": None,
                "FLOPs(G)": None,
                "Latency_stand": None,
                "Energy_stand": None
            }) for model in updated_model_list
        }

        config['model_list'] = updated_model_list
        config['model_info'] = updated_model_info

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    
    @staticmethod
    def load_config_and_update(script_dir, config_filename='config.json', model_dir='model/model_set/models'):
        config_path = os.path.join(script_dir, config_filename)
        base_path = os.path.join(script_dir, model_dir)
        
        # Update and load the configuration
        InitTools.update_config(config_path, base_path)
        with open(config_path, 'r') as file:
            return json.load(file)

