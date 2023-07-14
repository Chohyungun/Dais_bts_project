import pandas as pd
import matplotlib.pyplot as plt
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
from scipy import stats
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat)
from skimage.morphology import black_tophat, skeletonize, convex_hull_image  # noqa
from skimage.morphology import disk  # noqa
from skimage import data
from skimage.util import img_as_ubyte
from scipy.signal import wiener
from scipy.signal import convolve2d
from scipy import ndimage

"""
n (구조 요소의 크기): n은 구조 요소의 크기로, 필터링에 사용되는 주변 픽셀의 영역 크기를 결정함
n이 크면 더 넓은 영역의 정보를 사용하게 되어 잡음 제거 효과가 커질 수 있음
하지만 너무 큰 n값은 세부 사항을 잃어버릴 수 있으며, 이미지의 경계나 뾰족한 특징들이 흐려질 수 있음.
작은 n값은 잡음 제거 효과가 상대적으로 적을 수 있으며, 더 세부적인 특징을 보존할 수 있음

ENL (등가화된 개수): ENL은 등가화된 개수로, 잡음 모델링에 사용되는 매개변수
ENL이 클수록 잡음의 영향이 작아지며
이미지의 선명도가 향상될 수 있음
작은 ENL값은 잡음의 영향을 크게 받게 되어 이미지의 품질을 저하시킬 수 있음
ENL의 선택은 잡음의 성격과 원하는 결과에 따라 조정되어야 함
"""

def check_periodic_noise(img):
    # 이미지를 주파수 도메인으로 변환 (2D FFT)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    # 주파수 스펙트럼 시각화
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum')
    plt.colorbar()
    plt.show()

def check_noise(img):
    # 이미지를 1차원 배열로 변환
    pixels = img.flatten()

    # 픽셀값의 정규분포를 그리기 위한 코드
    plt.hist(pixels, bins=256, color='gray', alpha=0.9)
    plt.title('Pixel Value Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()



def LeeFilter(img, n, ENL, functionMode):
    # Local Mean function
    def localMean(img, size):
        kernel = np.ones((size, size))
        kernel_sum = np.sum(kernel)
        return np.divide(convolve2d(img, kernel, mode='same'), kernel_sum)

    # Local Variance function
    def localVariance(img, mean, size):
        kernel = np.ones((size, size))
        kernel_sum = np.sum(kernel)
        convolved_image = convolve2d(img ** 2, kernel, mode='same')
        return np.divide(convolved_image, kernel_sum) - mean ** 2

    # CFAR function
    def CFAR(k):
        return np.maximum(k - 1, 0)

    # Mask Detection function
    def maskDetect(img, size, mode):
        mask = np.ones_like(img)
        if mode == 'd':
            mask[:size, :] = np.array(-1).astype(np.uint8)
            mask[-size:, :] = np.array(-1).astype(np.uint8)
            mask[:, :size] = np.array(-1).astype(np.uint8)
            mask[:, -size:] = np.array(-1).astype(np.uint8)
        return mask

    # Local Mean with Mask function
    def localMeanMask(img, size, mask):
        kernel = np.ones((size, size))
        kernel_sum = np.sum(kernel)
        masked_image = np.multiply(img, mask)
        return np.divide(convolve2d(masked_image, kernel, mode='same'), kernel_sum)

    # Local Variance with Mask function
    def localVarianceMask(img, mean, size, mask):
        kernel = np.ones((size, size))
        kernel_sum = np.sum(kernel)
        masked_image = np.multiply(img ** 2, mask)
        return np.divide(convolve2d(masked_image, kernel, mode='same'), kernel_sum) - mean ** 2

    # Parameters
    dim = img.shape
    deadpixel = 0
    sigma_v = np.sqrt(1 / ENL)

    # Local values without mask
    E_norm = localMean(img, n)
    Var_y_norm = localVariance(img, E_norm, n)
    Var_x_norm = (Var_y_norm - E_norm ** 2 * sigma_v ** 2) / (1 + sigma_v ** 2)
    k_norm = Var_x_norm / Var_y_norm
    k_norm[k_norm < 0] = 0
    CFAR_lee = CFAR(k_norm)
    threshold = np.mean(CFAR_lee)

    # Local values with mask
    Mask = maskDetect(img, n, functionMode)
    Mask[k_norm < threshold] = np.array(-1).astype(np.uint8)
    E = localMeanMask(img, n, Mask)
    Var_y = localVarianceMask(img, E, n, Mask)

    # Weight
    Var_x = (Var_y - E ** 2 * sigma_v ** 2) / (1 + sigma_v ** 2)
    k = Var_x / Var_y
    k[k < 0] = 0

    # Creates the new picture
    img = E + k * (img - E)

    # Assign dead pixels
    img[dim[0] - n + 1:dim[0], :] = deadpixel
    img[0:n, :] = deadpixel
    img[:, dim[1] - n + 1:dim[1]] = deadpixel
    img[:, 0:n] = deadpixel

    return img


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
    
    # 미디안 필터
        """
        기본적으로 cv2.medianbulr와 하는건 같음 단, 커널사이즈를 cv2.medianblur는 홀수만 가능하지만
        ndimage.median_filter는 짝수도 가능함 따라서 짝수 커널 사이즈 할때 쓸 거임
        """
    elif preprocessing == 'ndmedian_filter':
        img = ndimage.median_filter(img, size=2)
        # img = ndimage.median_filter(img, ksize = 4)

    # 저주파 필터
    elif preprocessing == 'lpf':
        kernel = np.ones((5,5), np.float32) / 25
        img = cv2.filter2D(img, -1, kernel)
    
    elif preprocessing == 'gaussian_blur':
    # 가우시안 블러
        kernel = np.ones((5,5), np.float32) / 25
        img = cv2.GaussianBlur(img,(5,5),0)
    
    elif preprocessing == 'bilateral_filter':
    # 양방향 필터
        img =cv2.bilateralFilter(img,9,75,75)

    elif preprocessing == 'wiener_filter':
        std = np.std(img)
        img = wiener(img, (3, 3),std*0.005)

    elif preprocessing == 'erosion':
        """
        disk(radius) 함수는 반지름 radius를 가진 원형 구조 요소를 생성
        침식이나 팽창과 같은 모폴로지 연산에서 동네의 모양을 결정하는 데 사용
        예를 들어, 원형 구조 요소는 둥근 형태의 동네를 만들어 픽셀 주변의 영향을 정의

        disk 함수에서 사용한 반지름 값이 작아지면 구조 요소의 크기가 줄어들고, 반대로 반지름 값이 커지면 구조 요소의 크기가 커짐
        일반적으로 침식, 팽창 반지름은 3으로 지정
        """
        footprint = disk(3)
        img = erosion(img, footprint)
    
    elif preprocessing == 'dilation':
        footprint = disk(3)
        img = dilation(img, footprint)

    elif preprocessing == 'remove_high_outlier':
        threshold = 0.9995
        pixel_values = img.flatten()
        mean = np.mean(pixel_values)
        std = np.std(pixel_values)
        
        # 픽셀 값들의 정규분포에서 특정 값 이상의 이상치를 제거
        pixel_values[(pixel_values > mean + (threshold * std))] = mean

        # 이상치가 제거된 배열을 다시 이미지 형태로 변환
        img = pixel_values.reshape(img.shape)
    
        """
        비선형 Contra-Harmonic Mean 방법을 사용하여 이미지를 필터링 함.
        Contra-Harmonic Mean 필터는 산술 평균 필터보다 가우시안 유형의 잡음을 효과적으로 제거하고 에지 특징을 보존하는 비선형 평균 필터임
        Contra-Harmonic 필터는 P 값이 음수일 때 양의 이상치를, P 값이 양수일 때 음의 이상치를 효과적으로 제거하는 데 탁월함
        Q 값에 따라 양수 또는 음수의 이상치를 효과적으로 제거할 수 있음
        """

    elif preprocessing == 'contra_harmonic':
        # Contra-Harmonic Mean 필터링 적용
        mask_size = 1   # 마스크 크기 (홀수)
        Q = 2           # Q 파라미터

        img = np.zeros_like(img, dtype=np.float64)
        rows, cols = img.shape
        
        for i in range(rows):
            for j in range(cols):
                pixln = 0
                pixld = 0
                for m in range(-mask_size, mask_size + 1):
                    for n in range(-mask_size, mask_size + 1):
                        if (i + m >= 0 and i + m < rows and j + n >= 0 and j + n < cols and
                                mask_size + m >= 0 and mask_size + m < rows and
                                mask_size + n >= 0 and mask_size + n < cols):
                            
                            pixl1 = img[i + m, j + n] ** (Q + 1)
                            pixl2 = img[i + m, j + n] ** Q
                            pixln += pixl1
                            pixld += pixl2
                
                img[i, j] = pixln / pixld

    elif preprocessing == 'lee_filter':
        img = LeeFilter(img, 5, 100, 'd')
    
    elif preprocessing == 'periodic_filter':
        cutoff_freq = 1  # 주파수 스펙트럼에서 제거할 영역의 반경 (조정 가능)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)

        # 주파수 스펙트럼의 중심 좌표 계산
        rows, cols = img.shape
        crow, ccol = rows // 2, cols // 2

        # 필터링을 위해 주파수 스펙트럼의 특정 영역을 제거 (저주파 성분 제거)
        fshift[crow - cutoff_freq:crow + cutoff_freq, ccol - cutoff_freq:ccol + cutoff_freq] = 0

        # 주파수 스펙트럼을 다시 역변환하여 이미지로 변환
        f_ishift = np.fft.ifftshift(fshift)
        img = np.abs(np.fft.ifft2(f_ishift))
        img = 255 - img
    
    else:
        img = img
    img = cv2.resize(img, dsize = (cfg["target_size"][0], cfg["target_size"][1]), interpolation = cv2.INTER_CUBIC)
    img = img/255
    return img