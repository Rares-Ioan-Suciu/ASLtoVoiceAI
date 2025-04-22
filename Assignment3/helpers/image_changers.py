import random

import cv2
import numpy as np


# some image modifying function that can be used on the dataset to gather more data

def flip_img(img):
    return cv2.flip(img, 1)

def rotate_img(img):
    angle = random.randint(-45, 45)
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))

    return rotated


def zoom_img(img):
    zoom_factor = random.uniform(1.0, 1.3)
    h, w = img.shape[:2]
    new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
    resized = cv2.resize(img, (new_w, new_h))

    startx = new_w // 2 - w // 2
    starty = new_h // 2 - h // 2
    return resized[starty:starty + h, startx:startx + w]

def translation_img(img):
    h, w = img.shape[:2]
    max_trans_x = w * 0.2
    max_trans_y = h * 0.2
    tx = random.uniform(-max_trans_x, max_trans_x)
    ty = random.uniform(-max_trans_y, max_trans_y)
    trans_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, trans_matrix, (w, h))

def brightness_img(img):
    alpha = random.uniform(0.8, 1.3)  # contrast
    beta = random.randint(-30, 30)    # brightness
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def blurred_img(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

def noise_img(img):
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    noisy_img = cv2.add(img, noise)
    return noisy_img

def cutout_img(img):
    h, w = img.shape[:2]
    mask_size = random.randint(30, 60)
    x = random.randint(0, w - mask_size)
    y = random.randint(0, h - mask_size)
    img[y:y+mask_size, x:x+mask_size] = 0
    return img