import random
import numpy as np
from tqdm import tqdm
from scipy.cluster.vq import kmeans

from read_voc import VOCDataSet
from yolo_kmeans import k_means, wh_iou


def anchor_fitness(k: np.ndarray, wh: np.ndarray, thr: float):  # mutation fitness
    r = wh[:, None] / k[None]
    x = np.minimum(r, 1. / r).min(2)  # ratio metric
    # x = wh_iou(wh, k)  # iou metric
    best = x.max(1)
    f = (best * (best > thr).astype(np.float32)).mean()  # fitness
    bpr = (best > thr).astype(np.float32).mean()  # best possible recall
    return f, bpr

f, bpr = anchor_fitness(k, wh, thr)