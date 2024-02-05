#!/usr/nbin/env python3

import numpy as np 
import cv2

class canny:
    def __init__ (self):
        pass

    def gaussian_kernel(self,size,sigma=1.0):
        size = int(size)//2
        x,y = np.mgrid[-size:size+1,-size:size+1]
        normal = 1/(2.0 *np.pi *sigma**2)
        g= np.exp(-((x**2 + y**2)/(2.0*sigma**2)))* normal

        return g
    
if __name__ == '__main__':
    img = cv2.imread('edge-detectors/images/1.jpg')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow('img',gray_img)
    cv2.waitKey(0)