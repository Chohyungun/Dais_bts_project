import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import sys
import os
import yaml
import numpy as np
import random
import cv2
from sklearn.model_selection import train_test_split
from natsort import natsorted
from tqdm import tqdm
import imgaug.augmenters as iaa
import cfg
config = cfg.cfg

def basic_normalization(img):
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img

def spatial_normalization(img, min_value=0, max_value=255):
    np.set_printoptions(linewidth=np.inf)
    min_val, max_val, _, _ = cv2.minMaxLoc(img)
    img = (img - min_val) * (max_value - min_value) / (max_val - min_val) + min_value
    img = img.astype(np.float32)
    return img

def preprocess_img(img, preprocessing):
    if preprocessing == 'median_blur':
        img = cv2.medianBlur(img, ksize = 3)
    elif preprocessing == 'noise_drop':
        img = iaa.Dropout(p=(0, 0.2))(images = img).astype("uint8")
    elif preprocessing == 'his_equalized':
        img = cv2.equalizeHist(img)
    elif preprocessing == 'sobel_masking_y':
        img = cv2.Sobel(img, -1, 0, 1, delta = 128)
    elif preprocessing == 'scharr':
        img = cv2.Scharr(img, -1, 0, 1, delta=128)
    elif preprocessing == 'spatial_normalization':
        img = spatial_normalization(img)
    elif preprocessing == 'clahe':
        img = cv2.convertScaleAbs(img)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(3,3))
        img = clahe.apply(img)
    else:
        img = img
    img = cv2.resize(img, dsize = (cfg["target_size"][0], cfg["target_size"][1]), interpolation = cv2.INTER_CUBIC)
    img = img/255
    return img
