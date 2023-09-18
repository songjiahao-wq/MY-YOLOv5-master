import cv2
import numpy as np
import os

# 文件夹路径
folder_path = r'F:\yolov5-master-Insulator\Insulator_data\images\val_'
output_folder = r'F:\yolov5-master-Insulator\Insulator_data\images\val_npy'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# 1. 获取文件夹中的所有文件名
file_names = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

for file_name in file_names:
    # 2. 使用OpenCV读取图像
    image_path = os.path.join(folder_path, file_name)
    image = cv2.imread(image_path)

    # 3. 使用numpy.save保存图像
    npy_file_name = os.path.splitext(file_name)[0] + '.npy'  # 将图像的扩展名更改为.npy
    npy_path = os.path.join(output_folder, npy_file_name)
    np.save(npy_path, image)
