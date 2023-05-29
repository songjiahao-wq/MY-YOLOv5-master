import torch
import cv2
import json
import torch
import sys
import numpy as np
import cv2
from pathlib import Path
from models.common import DetectMultiBackend
# from ensemble_boxes import weighted_boxes_fusion
import os
from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.general import check_img_size, non_max_suppression, scale_boxes,check_img_size
from utils.augmentations import letterbox
# # Load Yolov5 model
# model_path = r'weights/yolov5s.pt'
# data = r'data/coco128.yaml'
# device = torch.device("cuda:0")
# @torch.no_grad()
# def init():
#     weights = model_path
#     device = 'cuda:0'  # cuda device, i.e. 0 or 0,1,2,3 or
#
#     half = True  # use FP16 half-precision inference
#     device = select_device(device)
#     w = str(weights[0] if isinstance(weights, list) else weights)
#     # model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, device=device)
#     model = DetectMultiBackend(w, device=device, dnn=False, data=data,fp16=half)
#
#     model.eval()
#     return model
# # model = torch.load(r'D:\yanyi\xianyu\1\facedetector_facemask3\facedetector_facemask3\untitled\weights\yolo.pt', map_location=torch.device('cuda'))
# model = init()
# conf_thres = 0.3 # confidence threshold
# iou_thres = 0.05  # NMS IOU threshold
# classes = int(0)
# max_det = 1000  # maximum detections per image
# # Define input image path
# img_path = r'D:\yanyi\myGithub\MY-YOLOv5-master\data\images\bus.jpg'
#
# # Load input image
# im = cv2.imread(img_path)
# stride = 32
# imgsz = [640,640]
# imgsz = check_img_size(imgsz, s=stride)  # check image size
# # Process image here
# fake_result = {}
# fake_result["model_data"] = {"objects": []}
#
# img = letterbox(im, imgsz)[0]
# print(img.shape)
# img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
# img = np.ascontiguousarray(img)
# img = torch.from_numpy(img).to(device)
# img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
# img /= 255  # 0 - 255 to 0.0 - 1.0
# if len(img.shape) == 3:
#     img = img[None]  # expand for batch dim
# names = {
#         0: 'person',
#         1: 'stripe_spot',
#         2: 'single_spot',
#         3: 'multi_spot'
#     }
#
# # Get predictions
# pred = model(img)
# pred = non_max_suppression(pred, conf_thres, iou_thres, classes, False, max_det=max_det)
# re_boxes = list()
# re_confidence = list()
# re_classes = list()
# re_mask_id = list()
# for i, det in enumerate(pred):  # per image
#     det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im.shape).round()
#     for *xyxy, conf, cls in reversed(det):
#         xyxy_list = torch.tensor(xyxy).view(1, 4).view(-1).tolist()
#         conf_list = conf.tolist()
#         label = names[int(cls)]
#         re_boxes.append([int(xyxy_list[0]), int(xyxy_list[1]), int(xyxy_list[2]), int(xyxy_list[3])])
#         re_confidence.append(conf_list)
#         re_classes.append('face')
#         re_mask_id.append(int(cls))
#
# # Print the pred
# print("Bounding Boxes:\n", re_boxes)
# print("Confidences:\n", re_confidence)
# print("Classes:\n", re_classes)
# print("Mask IDs:\n", re_mask_id)

def detctapi():
    # Load Yolov5 model
    model_path = r'../weights/yolov5s.pt'
    data = r'data/coco128.yaml'
    device = torch.device("cuda:0")

    @torch.no_grad()
    def init():
        weights = model_path
        device = 'cuda:0'  # cuda device, i.e. 0 or 0,1,2,3 or

        half = True  # use FP16 half-precision inference
        device = select_device(device)
        w = str(weights[0] if isinstance(weights, list) else weights)
        # model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, device=device)
        model = DetectMultiBackend(w, device=device, dnn=False, data=data, fp16=half)

        model.eval()
        return model

    # model = torch.load(r'D:\yanyi\xianyu\1\facedetector_facemask3\facedetector_facemask3\untitled\weights\yolo.pt', map_location=torch.device('cuda'))
    model = init()
    conf_thres = 0.3  # confidence threshold
    iou_thres = 0.05  # NMS IOU threshold
    classes = int(0)
    max_det = 1000  # maximum detections per image
    img_path = r'/data/images/bus.jpg'

    # Load input image
    im = cv2.imread(img_path)

    # Load input image
    stride = 32
    imgsz = [640, 640]
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # Process image here
    fake_result = {}
    fake_result["model_data"] = {"objects": []}

    img = letterbox(im, imgsz)[0]
    print(img.shape)
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
    img /= 255  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    names = {
        0: 'person',
        1: 'stripe_spot',
        2: 'single_spot',
        3: 'multi_spot'
    }

    # Get predictions
    pred = model(img)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, False, max_det=max_det)
    re_boxes = list()
    re_confidence = list()
    re_classes = list()
    re_mask_id = list()
    for i, det in enumerate(pred):  # per image
        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im.shape).round()
        for *xyxy, conf, cls in reversed(det):
            xyxy_list = torch.tensor(xyxy).view(1, 4).view(-1).tolist()
            conf_list = conf.tolist()
            label = names[int(cls)]
            re_boxes.append([int(xyxy_list[0]), int(xyxy_list[1]), int(xyxy_list[2]), int(xyxy_list[3])])
            re_confidence.append(conf_list)
            re_classes.append('face')
            re_mask_id.append(int(cls))
    print("Bounding Boxes:\n", re_boxes)
    print("Confidences:\n", re_confidence)
    print("Classes:\n", re_classes)
    print("Mask IDs:\n", re_mask_id)
    # return re_boxes, re_confidence, re_classes, re_mask_id
detctapi()
def process_image(handle=None, input_image=None, args=None, **kwargs):
    half = False  # use FP16 half-precision inference
    conf_thres = 0.3 # confidence threshold
    iou_thres = 0.05  # NMS IOU threshold

    max_det = 1000  # maximum detections per image
    imgsz = [640,640]
    names = {
        0: 'photovoltaic_panels',
        1: 'stripe_spot',
        2: 'single_spot',
        3: 'multi_spot'
    }

    stride = 32
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # Process image here
    fake_result = {}
    fake_result["model_data"] = {"objects": []}

    img = letterbox(input_image, imgsz, stride, False)[0]
    print(img.shape)
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if handle.fp16 else img.float()  # uint8 to fp16/32
    img /= 255  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    pred = handle(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=max_det)

    for i, det in enumerate(pred):  # per image
        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], input_image.shape).round()
        for *xyxy, conf, cls in reversed(det):
            xyxy_list = torch.tensor(xyxy).view(1, 4).view(-1).tolist()
            conf_list = conf.tolist()
            label = names[int(cls)]
            fake_result['model_data']['objects'].append({
                "xmin": int(xyxy_list[0]),
                "ymin": int(xyxy_list[1]),
                "xmax": int(xyxy_list[2]),
                "ymax": int(xyxy_list[3]),
                "confidence": conf_list,
                "name": label
                })
    return json.dumps(fake_result, indent=4)