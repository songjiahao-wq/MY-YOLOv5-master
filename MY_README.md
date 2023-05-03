<details open>
<summary>Install</summary>
通过timm库更换骨干网：

替换Backbone网络：train_custom.py、models/backbone_main.py、models/yolo_Backbone.py、models/yolov5-custom.yaml

```bash
models/main_backbone.py：输出timm已支持骨干，以及测试骨干网是否可使用
models/yolo_Backbone.py：yolo的Backbone版本
models/yolov5-custom.yaml：自定义的骨干网络yaml
train_custom.py：训练自定义骨干网
```
冻结训练：添加从后向前冻结训练
```bash
python train.py --Freeze -3#冻结后三层
```
待更新：（轻量化网络名字）
</details>

