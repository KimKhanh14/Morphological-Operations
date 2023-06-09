import sys
import cv2
import getopt
import numpy as np
from morphological_operator import binary
from morphological_operator import grayscale


def operator(in_file, out_file, mor_op, wait_key_time=0):
    img_origin = cv2.imread(in_file)
    cv2.imshow('original image', img_origin)
    cv2.waitKey(wait_key_time)

    img_gray = cv2.imread(in_file, 0)
    cv2.imshow('gray image', img_gray)
    cv2.waitKey(wait_key_time)

    (thresh, img) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow('binary image', img)
    cv2.waitKey(wait_key_time)

    kernel = np.ones((3, 3), np.uint8)
    img_out = None

    if mor_op == 'dilation':
        img_dilation = cv2.dilate(img, kernel)
        cv2.imshow('OpenCV dilation image', img_dilation)
        cv2.waitKey(wait_key_time)

        img_dilation_manual = binary.dilate(img, kernel)
        cv2.imshow('manual dilation image', img_dilation_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_dilation_manual
    elif mor_op == 'erosion':
        img_erosion = cv2.erode(img, kernel)
        cv2.imshow('OpenCV erosion image', img_erosion)
        cv2.waitKey(wait_key_time)

        img_erosion_manual = binary.erode(img, kernel)
        cv2.imshow('manual erosion image', img_erosion_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_erosion_manual
    elif mor_op == 'opening':
        img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        cv2.imshow('OpenCV opening image', img_opening)
        cv2.waitKey(wait_key_time)

        img_opening_manual = binary.opening(img, kernel)
        cv2.imshow('manual opening image', img_opening_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_opening_manual
    elif mor_op == 'closing':
        img_closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('OpenCV closing image', img_closing)
        cv2.waitKey(wait_key_time)

        img_closing_manual = binary.closing(img, kernel)
        cv2.imshow('manual closing image', img_closing_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_closing_manual
    elif mor_op == 'hitmiss':
        kernel = np.array((
            [1,1,1],
            [0,1,-1],
            [0,1,-1]), dtype = "int")

        img_hitmiss = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel)
        cv2.imshow('OpenCV hit or miss image', img_hitmiss)
        cv2.waitKey(wait_key_time)

        img_hitmiss_manual = binary.hitmiss(img, kernel)
        cv2.imshow('manual hit or miss image', img_hitmiss_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_hitmiss_manual
    elif mor_op == 'thinning':
        img_thinning = cv2.ximgproc.thinning(img)
        cv2.imshow('OpenCV thinning image', img_thinning)
        cv2.waitKey(wait_key_time)

        img_thinning_manual = binary.thinning(img,kernel)
        cv2.imshow('manual thinning image', img_thinning_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_thinning_manual
    elif mor_op == 'gradient':
        img_gradient = cv2.morphologyEx(img_gray, cv2.MORPH_GRADIENT,kernel)
        cv2.imshow('OpenCV morphological gradient image', img_gradient)
        cv2.waitKey(wait_key_time)

        img_out = img_gradient
    elif mor_op == 'tophat':
        img_tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
        cv2.imshow('OpenCV Top-Hat image', img_tophat)
        cv2.waitKey(wait_key_time)

        img_out = img_tophat
    elif mor_op == 'blackhat':
        img_blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
        cv2.imshow('OpenCV Black-Hat image', img_blackhat)
        cv2.waitKey(wait_key_time)

        img_out = img_blackhat

    if img_out is not None:
        cv2.imwrite(out_file, img_out)


def main(argv):
    input_file = ''
    output_file = ''
    mor_op = ''
    wait_key_time = 0

    description = 'main.py -i <input_file> -o <output_file> -p <mor_operator> -t <wait_key_time>'

    try:
        opts, args = getopt.getopt(argv, "hi:o:p:t:", ["in_file=", "out_file=", "mor_operator=", "wait_key_time="])
    except getopt.GetoptError:
        print(description)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(description)
            sys.exit()
        elif opt in ("-i", "--in_file"):
            input_file = arg
        elif opt in ("-o", "--out_file"):
            output_file = arg
        elif opt in ("-p", "--mor_operator"):
            mor_op = arg
        elif opt in ("-t", "--wait_key_time"):
            wait_key_time = int(arg)

    print('Input file is ', input_file)
    print('Output file is ', output_file)
    print('Morphological operator is ', mor_op)
    print('Wait key time is ', wait_key_time)

    operator(input_file, output_file, mor_op, wait_key_time)
    cv2.waitKey(wait_key_time)


if __name__ == "__main__":
    main(sys.argv[1:])
