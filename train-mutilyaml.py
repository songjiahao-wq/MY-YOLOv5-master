import os
import subprocess

config_directory = r'D:\my_job\code\xianyu\沈阳理工大学本科code\cfg\Lightmodel\1'
config_files = os.listdir(config_directory)

for config_file in config_files:
    if config_file.endswith('.yaml'):
        config_path = os.path.join(config_directory, config_file)
        print(f"Training model with config: {config_path}")
        subprocess.run(['python', 'train.py', '--cfg', config_path])
