import os
import subprocess
#运行的时候需要修改yolo里Detect/foward的输出
# Change the working directory to 'run/'
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

config_files = r'./models\yolov5s.yaml'
print(f"Training model with config: {config_files}")
subprocess.run(['python', 'train.py', '--cfg', config_files,\
                '--data', 'data/coco128.yaml',\
                '--epoch', '100', '--batch-size', '2'
                ])