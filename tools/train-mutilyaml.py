import os
import subprocess
#运行的时候需要修改yolo里Detect/foward的输出
# Change the working directory to 'run/'
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

config_directory = r'D:\my_job\code\xianyu\沈阳理工大学本科code\cfg\Lightmodel\1'
config_files = os.listdir(config_directory)

for config_file in config_files:
    if config_file.endswith('.yaml'):
        config_path = os.path.join(config_directory, config_file)
        print(f"Training model with config: {config_path}")
        subprocess.run(['python', 'train.py', '--cfg', config_path])
