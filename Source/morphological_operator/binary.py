from ast import And
from email.mime import image
import numpy as np
import cv2

def erode(img, kernel):
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    kernel_ones_count = kernel.sum()
    eroded_img = np.zeros((img.shape[0] + kernel.shape[0] - 1, img.shape[1] + kernel.shape[1] - 1))
    img_shape = img.shape

    x_append = np.zeros((img.shape[0], kernel.shape[1] - 1))
    img = np.append(img, x_append, axis=1)

    y_append = np.zeros((kernel.shape[0] - 1, img.shape[1]))
    img = np.append(img, y_append, axis=0)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            i_ = i + kernel.shape[0]
            j_ = j + kernel.shape[1]
            temp = (kernel * img[i:i_, j:j_]).sum()
            if(temp > 255): temp = temp/255
            if kernel_ones_count == temp:
                eroded_img[i + kernel_center[0], j + kernel_center[1]] = 1

    return eroded_img[:img_shape[0], :img_shape[1]]

def dilate(img, kernel):
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    dilate_img = np.ones((img.shape[0] + kernel.shape[0] - 1, img.shape[1] + kernel.shape[1] - 1))
    img_shape = img.shape

    x_append = np.zeros((img.shape[0], kernel.shape[1] - 1))
    img = np.append(img, x_append, axis=1)

    y_append = np.zeros((kernel.shape[0] - 1, img.shape[1]))
    img = np.append(img, y_append, axis=0)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            i_ = i + kernel.shape[0]
            j_ = j + kernel.shape[1]
            temp = (kernel * img[i:i_, j:j_]).sum()
            if(temp > 255): temp = temp/255
            if 0 == temp:
                dilate_img[i + kernel_center[0], j + kernel_center[1]] = 0

    return dilate_img[:img_shape[0], :img_shape[1]]

def opening(img, kernel):
    return dilate(erode(img,kernel),kernel)

def closing(img, kernel):
    return erode(dilate(img,kernel),kernel)

def hitmiss(img, kernel):
    b1 = kernel
    b2 = kernel
    b1 = np.where(b1 == -1, 0, b1)
    b2 = np.where(b2 == 1, 0, b2)
    b2 = np.where(b2 == -1, 1, b2)
    a = erode(img,b1)
    b = erode(~(np.where(img == 255, 1, img)), b2)
    hitmiss_img = np.logical_and(a,b)
    return 1.0*hitmiss_img

def thinning(img,kernel):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    b1 = kernel
    b2 = kernel
    b1 = np.where(b1 == -1, 0, b1)
    b2 = np.where(b2 == 1, 0, b2)
    b2 = np.where(b2 == -1, 1, b2)
    a = erode(img,b1)
    b = erode(~(np.where(img == 255, 1, img)), b2)
    hitmiss_img = np.logical_and(a,b)
    thin_img = np.logical_and(img,~(hitmiss_img))
    return 1.0*thin_img