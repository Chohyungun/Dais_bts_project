import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import yaml
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from natsort import natsorted
from tqdm import tqdm
from preprocessing import spatial_normalization
import cfg
config = cfg.cfg


def get_weld_image(cfg, img_path, margin_pixel):
    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    original_shape = img.shape[1]
    img = spatial_normalization(img)
    img = cv2.resize(img, (cfg["target_size"][0], cfg["target_size"][1]))
    image = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    image = np.array(image, dtype=np.float32)

    mean_list = []
    for i in range(cfg["target_size"][0]):
        mean_list.append(np.mean(image[:,i]))

    image = np.gradient(np.squeeze(mean_list))
    y1, y2 = cfg["target_size"][0] - int(np.argmax(image)* cfg["target_size"][0] / cfg["target_size"][0]), cfg["target_size"][0] - int(np.argmin(image) * cfg["target_size"][0] / cfg["target_size"][0]) 

    weld_margin = int((512 * margin_pixel) / original_shape)
    weld = (y1 - weld_margin, y2 + weld_margin)

    result = img[weld[0] - weld_margin : weld[1] + weld_margin]

    norm = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    h, w = norm.shape
    width,height = cfg["target_size"][0], cfg["target_size"][1]
    dst = 255 * np.ones((height, width), dtype=np.uint8)
    roi = result[0:h, 0:w]
    dst[int((cfg["target_size"][0])/2 - h/2):int((cfg["target_size"][1])/2 + h/2), 0:cfg["target_size"][0]] = roi
    return dst