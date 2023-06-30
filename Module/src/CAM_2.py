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
from weld_extract import get_weld_image
import preprocessing as pp
import cfg
config = cfg.cfg

img = cv2.imread("/DaiS_internship/Module/src/sp.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, dsize = (config["target_size"][0], config["target_size"][1]), interpolation = cv2.INTER_CUBIC)
model_path = config["model_path"]
model = tf.keras.models.load_model(model_path)

def make_CAM(model,img):
    img = np.expand_dims(img, axis=0)
    x_data = np.array(img)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.layers[-5].output, model.output]
    )

    # Grad-CAM 계산을 위한 그래디언트 함수
    class_idx = 0
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(x_data)
        loss = predictions[:, class_idx]

    output = conv_output[0]
    grads = tape.gradient(loss, conv_output)[0]

    # 특성 맵 채널별 가중치 계산
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.ones(output.shape[0:2], dtype=np.float32)


    # 가중합으로 Class Activation Map 계산
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]


    # CAM을 원본 이미지 크기로 업샘플링
    cam = cv2.resize(cam.numpy(), (config["target_size"][0],  config["target_size"][1]))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)
    plt.imshow(heatmap, cmap='jet')


make_CAM(model, img)