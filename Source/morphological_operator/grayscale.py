from ast import And
from email.mime import image
import numpy as np
import cv2

def dilation(img,kernel):
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    dilate_img = 255*np.ones((img.shape[0] + kernel.shape[0] - 1, img.shape[1] + kernel.shape[1] - 1))
    img_shape = img.shape

    x_append = 255*np.ones((img.shape[0], kernel.shape[1] - 1))
    img = np.append(img, x_append, axis=1)

    y_append = 255*np.ones((kernel.shape[0] - 1, img.shape[1]))
    img = np.append(img, y_append, axis=0)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            i_ = i + kernel.shape[0]
            j_ = j + kernel.shape[1]
            #temp = (kernel * img[i:i_, j:j_]).sum()
            #if(temp > 255): temp = temp/255
            #if 0 == temp:
            dilate_img[i + kernel_center[0], j + kernel_center[1]] = np.min(kernel * img[i:i_, j:j_])

    return dilate_img[:img_shape[0], :img_shape[1]]