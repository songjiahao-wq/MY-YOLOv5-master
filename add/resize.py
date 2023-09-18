"""
YOLO的自适应改变图片机制，可以预先改变图片的大小
"""
import os.path

import cv2
import numpy as np
import glob
import math

def load_image(im, img_size=640, augment=False):
    # Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)
    assert im is not None, f'Image Not Found'
    h0, w0 = im.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # ratio
    if r != 1:  # if sizes are not equal
        interp = cv2.INTER_LINEAR if (augment or r > 1) else cv2.INTER_AREA
        im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
    return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    # 计算需要填充的边的像素
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)
# 图片源路径
img_path = r'F:\yolov5-master-Insulator\Insulator_data\images\train_5s/*.jpg'
# 复制文件总路径
src_root = r'F:\yolov5-master-Insulator\Insulator_data\images\train/'
if not os.path.exists(src_root):
    os.makedirs(src_root)
img_list = glob.glob(img_path)
img_number = 0
for img_name in img_list:
    img_number +=1
    print(img_number)
    img = cv2.imread(img_name)
    print(img.shape)
    img2 = load_image(img)[0]
    print(img2.shape)
    # print(src_root+os.path.basename(img_name))
    cv2.imwrite(src_root+os.path.basename(img_name),img2)