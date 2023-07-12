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
<details open>
<summary>Custom Backbone</summary>

```bash
# train
tools/train_custom.py
# yolo
models/yolo_Backbone.py
# yaml
cfg/Lightmodel/yolov5-vanillaNet.yaml
#
```
</details>
<details open>
<summary>生成热力图方法</summary>
方法1

```bash
# 
python tools/yolov5_heatmap.py
#
```
方法2
```bash
# 
python tools/main_gram.py
#
```
</details>
<details open>
<summary>添加dyhead</summary>

```bash
cfg/yolov5s-Improved/head_improved/yolov5s-dyhead.yaml
```
</details>

<details open>
<summary>论文复现及改动：特征融合网络ES-Net.py</summary>

```bash
cfg/yolov5s-Improved/head_improved/yolov5s-dyhead.yaml
```
</details>