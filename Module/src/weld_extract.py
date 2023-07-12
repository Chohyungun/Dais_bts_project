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


def get_weld_image(config, img_path):

    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    img = spatial_normalization(img)
    img = cv2.resize(img, (512, 512))
    image = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    image = np.array(image, dtype=np.float32)

    mean_list = []
    for i in range(config["target_size"][0]):
        mean_list.append(np.mean(image[:,i]))

    image = np.gradient(np.squeeze(mean_list))
    y1, y2 = config["target_size"][0] - int(np.argmax(image)* config["target_size"][0] / config["target_size"][0]), config["target_size"][0] - int(np.argmin(image) * config["target_size"][0] / config["target_size"][0]) 

    weld = (y1, y2)
    result = img[weld[0] - 15 : weld[1] + 15]

    norm = spatial_normalization(result)

    h, w = norm.shape
    width,height = config["target_size"][0], config["target_size"][1]
    dst = 255 * np.ones((height, width), dtype=np.uint8)
    roi = result[0:h, 0:w]
    dst[int((config["target_size"][0]/2)-h/2):int((config["target_size"][1])/2+h/2), 0:config["target_size"][0]] = roi

    return dst
