import cv2 as cv
from pathlib import Path


def batch_resize(img_dir, dest_img_dir, size=640):
    img_dir = Path(img_dir)
    dest_img_dir = Path(dest_img_dir)
    dest_img_dir.mkdir(parents=True, exist_ok=True)

    for img_path in img_dir.glob('*.jpg'):
        img = cv.imread(str(img_path))
        h, w = img.shape[:2]
        ratio = size / max(h, w)
        img = cv.resize(img, None, fx=ratio, fy=ratio, interpolation=cv.INTER_LINEAR)
        cv.imwrite(str(dest_img_dir / img_path.name), img)


batch_resize(r'F:\sjh\xianyu\xianyu_7000\dataset\newyolo\images\train-ori',
             r'F:\sjh\xianyu\xianyu_7000\dataset\newyolo\images\train640')
# batch_resize(r'D:\workPython\yolov5-6.2\data\earplug_data\images\val',
#              r'D:\workPython\yolov5-6.2\data\earplug_data\images_mini\val')