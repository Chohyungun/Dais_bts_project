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

def make_CAM(df, config, preprocessing):
    img_list = []
    margin_pixel = config["margin"]
    for what in range(len(df)):
        for i in range(len(df)):
            weld_type = config["weld_type"]
            IMG_NAME = df["path"].iloc[i]
            if weld_type == True:
                img = get_weld_image(config, IMG_NAME,margin_pixel)
                img = pp.preprocess_img(img,preprocessing)
                img = np.expand_dims(img, axis=0)
                img_list.append(img)
            else:
                img = cv2.imread(IMG_NAME,cv2.IMREAD_GRAYSCALE)
                img = pp.spatial_normalization(img)
                img = pp.preprocess_img(img,preprocessing)
                img = np.expand_dims(img, axis=0)
                img_list.append(img)
        x_data = np.array(img_list)

        # 가중치 파일 로드
        model_path = config["model_path"]
        model = tf.keras.models.load_model(model_path)
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.layers[-5].output, model.output]
        )

        # Grad-CAM 계산을 위한 그래디언트 함수
        class_idx = 0
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(x_data[what])
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
        RAW_IMG_NAME = df["path"].iloc[what]
        raw_img = pp.load_img(RAW_IMG_NAME,"none")
        cam = cv2.resize(cam.numpy(), (cfg["target_size"][0],  cfg["target_size"][1]))
        cam = np.maximum(cam, 0)
        heatmap = cam / np.max(cam)
        fit, axes = plt.subplots(1, 2, figsize = (18, 5))

        axes[0].set_title('RAW')
        axes[0].imshow(raw_img, cmap='gray')

        axes[1].set_title('CAM')
        axes[1].imshow(heatmap, cmap='jet')
        save_path = cfg["save_path"]
        try:
            result = os.path.join(save_path, preprocessing)
            os.makedirs(result)
        except:
            pass
        result_fig = os.path.join(result, df["path"].iloc[what].split("/")[-1] )
        plt.savefig(result_fig)
        plt.close()