#!/usr/nbin/env python3

import numpy as np 
import cv2

class canny:
    pass
if __name__ == '__main__':
    img = cv2.imread('edge-detectors/images/1.jpg')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow('img',gray_img)
    cv2.waitKey(0)