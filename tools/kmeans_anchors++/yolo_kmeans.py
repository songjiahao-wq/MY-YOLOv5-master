import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import glob
import xml.etree.ElementTree as ET

# 这里IOU的概念更像是只是考虑anchor的长宽
def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = np.minimum(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


# 计算单独一个点和一个中心的距离
def single_distance(center, point):
    center_x, center_y = center[0] / 2, center[1] / 2
    point_x, point_y = point[0] / 2, point[1] / 2
    return np.sqrt((center_x - point_x) ** 2 + (center_y - point_y) ** 2)


# 计算中心点和其他点直接的距离
def calc_distance(boxes, clusters):
    """
    :param obs: 所有的观测点
    :param clusters: 中心点
    :return:每个点对应中心点的距离
    """
    distances = []
    for box in boxes:
        # center_x, center_y = x/2, y/2
        distance = []
        for center in clusters:
            # center_xc, cneter_yc = xc/2, yc/2
            distance.append(single_distance(box, center))
        distances.append(distance)

    return distances


# k_means++计算中心坐标
def calc_center(boxes, k):
    box_number = boxes.shape[0]
    # 随机选取第一个中心点
    first_index = np.random.choice(box_number, size=1)
    clusters = boxes[first_index]
    # 计算每个样本距中心点的距离
    dist_note = np.zeros(box_number)
    dist_note += np.inf
    for i in range(k):
        # 如果已经找够了聚类中心，则退出
        if i + 1 == k:
            break
        # 计算当前中心点和其他点的距离
        for j in range(box_number):
            j_dist = single_distance(boxes[j], clusters[i])
            if j_dist < dist_note[j]:
                dist_note[j] = j_dist
        # 转换为概率
        dist_p = dist_note / dist_note.sum()
        # 使用赌轮盘法选择下一个点
        next_index = np.random.choice(box_number, 1, p=dist_p)
        next_center = boxes[next_index]
        clusters = np.vstack([clusters, next_center])
    return clusters


# k-means聚类，且评价指标采用IOU
def k_means(boxes, k, dist=np.median, use_iou=True, use_pp=True):
    """
    yolo k-means methods
    Args:
        boxes: 需要聚类的bboxes,bboxes为n*2包含w，h
        k: 簇数(聚成几类)
        dist: 更新簇坐标的方法(默认使用中位数，比均值效果略好)
        use_iou：是否使用IOU做为计算
        use_pp：是否是同k-means++算法
    """
    box_number = boxes.shape[0]
    last_nearest = np.zeros((box_number,))
    # 在所有的bboxes中随机挑选k个作为簇的中心
    if not use_pp:
        clusters = boxes[np.random.choice(box_number, k, replace=False)]
    # k_means++计算初始值
    else:
        clusters = calc_center(boxes, k)

    # print(clusters)
    while True:
        # 计算每个bboxes离每个簇的距离 1-IOU(bboxes, anchors)
        if use_iou:
            distances = 1 - wh_iou(boxes, clusters)
        else:
            distances = calc_distance(boxes, clusters)
        # 计算每个bboxes距离最近的簇中心
        current_nearest = np.argmin(distances, axis=1)
        # 每个簇中元素不在发生变化说明以及聚类完毕
        if (last_nearest == current_nearest).all():
            break  # clusters won't change
        for cluster in range(k):
            # 根据每个簇中的bboxes重新计算簇中心
            clusters[cluster] = dist(boxes[current_nearest == cluster], axis=0)

        last_nearest = current_nearest

    return clusters , current_nearest
def load_data(path):
    data = []
    # -------------------------------------------------------------#
    #   对于每一个xml都寻找box
    # -------------------------------------------------------------#
    for xml_file in tqdm(glob.glob('{}/*xml'.format(path))):
        tree = ET.parse(xml_file)
        height = int(tree.findtext('./size/height'))
        width = int(tree.findtext('./size/width'))
        if height <= 0 or width <= 0:
            continue

        # -------------------------------------------------------------#
        #   对于每一个目标都获得它的宽高
        # -------------------------------------------------------------#
        for obj in tree.iter('object'):
            xmin = int(float(obj.findtext('bndbox/xmin'))) / width
            ymin = int(float(obj.findtext('bndbox/ymin'))) / height
            xmax = int(float(obj.findtext('bndbox/xmax'))) / width
            ymax = int(float(obj.findtext('bndbox/ymax'))) / height

            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)
            # 得到宽高
            data.append([xmax - xmin, ymax - ymin])
    return np.array(data)
def cas_ratio(box, cluster):
    ratios_of_box_cluster = box / cluster
    ratios_of_cluster_box = cluster / box
    ratios = np.concatenate([ratios_of_box_cluster, ratios_of_cluster_box], axis=-1)

    return np.max(ratios, -1)


def avg_ratio(box, cluster):
    return np.mean([np.min(cas_ratio(box[i], cluster)) for i in range(box.shape[0])])

if __name__ == '__main__':
    np.random.seed(0)
    # -------------------------------------------------------------#
    #   运行该程序会计算'./VOCdevkit/VOC2007/Annotations'的xml
    #   会生成yolo_anchors.txt
    # -------------------------------------------------------------#
    input_shape = [640, 640]
    anchors_num = 9
    # -------------------------------------------------------------#
    #   载入数据集，可以使用VOC的xml
    # -------------------------------------------------------------#
    path = 'D:\songjiahao\DATA\VOCdevkit\VOC2012\Annotations'

    # -------------------------------------------------------------#
    #   载入所有的xml
    #   存储格式为转化为比例后的width,height
    # -------------------------------------------------------------#
    print('Load xmls.')
    data = load_data(path)
    print('Load xmls done.')

    # -------------------------------------------------------------#
    #   使用k聚类算法
    # -------------------------------------------------------------#
    print('K-means boxes.')
    cluster, near = k_means(data, anchors_num)
    print('K-means boxes done.')
    data = data * np.array([input_shape[1], input_shape[0]])
    cluster = cluster * np.array([input_shape[1], input_shape[0]])

    # -------------------------------------------------------------#
    #   绘图
    # -------------------------------------------------------------#
    for j in range(anchors_num):
        plt.scatter(data[near == j][:, 0], data[near == j][:, 1])
        plt.scatter(cluster[j][0], cluster[j][1], marker='x', c='black')
    plt.savefig("kmeans_for_anchors.jpg")
    plt.show()
    print('Save kmeans_for_anchors.jpg in root dir.')

    cluster = cluster[np.argsort(cluster[:, 0] * cluster[:, 1])]
    print('avg_ratio:{:.2f}'.format(avg_ratio(data, cluster)))
    print(cluster)

    f = open("yolo_anchors.txt", 'w')
    row = np.shape(cluster)[0]
    for i in range(row):
        if i == 0:
            x_y = "%d,%d" % (cluster[i][0], cluster[i][1])
        else:
            x_y = ", %d,%d" % (cluster[i][0], cluster[i][1])
        f.write(x_y)
    f.close()