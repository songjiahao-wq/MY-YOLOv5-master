
## 生成聚类图
```bash
python tools/kmeans_anchors++/yolo_kmeans.py
```
<summary>效果输出</summary>

<div>
<p align="left"><img width="800" src="https://github.com/songjiahao-wq/MY-YOLOv5-master/blob/master/add/ScreenShots/img.png"></p>
</div>
在tools/kmeans_anchors++/yolo_kmeans.py中，实现了

    Args:
        boxes: 需要聚类的bboxes,bboxes为n*2包含w，h
        k: 簇数(聚成几类)
        dist: 更新簇坐标的方法(默认使用中位数，比均值效果略好)
        use_iou：是否使用IOU做为计算
        use_pp：是否是同k-means++算法

