#!/usr/nbin/env python3

import numpy as np 
import cv2
from scipy.ndimage import convolve

class canny:
    def __init__ (self, image, gaussian_kernel_size=5):
        self.kernel_size = gaussian_kernel_size #5x5 kernel
        self.gray_img = image
        self.gaussian = self.gaussian_kernel()
        self.lowthreshold=0.05
        self.highthreshold=0.15

    def gaussian_kernel(self,sigma=1.0):
        size = self.kernel_size
        size = int(size)//2
        x,y = np.mgrid[-size:size+1,-size:size+1]
        normal = 1/(2.0 *np.pi *sigma**2)
        g= np.exp(-((x**2 + y**2)/(2.0*sigma**2)))* normal

        return g
    
    def convolve(self,image, kernel):
        kernel_height, kernel_width = kernel.shape
        # Calculate padding width and height
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2
        
        # Pad the image with zeros on all sides
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
        padded_height, padded_width = padded_image.shape
       
        output = np.zeros_like(image)
        
        # Perform the convolution operation
        for y in range(pad_height, padded_height - pad_height):
            for x in range(pad_width, padded_width - pad_width):
                # Extract the current region of interest
                region = padded_image[y - pad_height:y + pad_height + 1, x - pad_width:x + pad_width + 1]
                # Perform element-wise multiplication and sum the result
                output[y - pad_height, x - pad_width] = np.sum(region * kernel)
                
        return output
    
    def sobel_filters(self, image):
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        Ix = self.convolve(image, Kx)
        Iy = self.convolve(image, Ky)
        G = np.sqrt(Ix**2 + Iy**2)
        theta = np.arctan2(Iy, Ix)
        return G, theta
    
    def non_max_suppression(self,gradient, theta):
        M, N = gradient.shape
        Z = np.zeros((M,N), dtype=np.float32)
        angle = theta * 180. / np.pi
        angle[angle < 0] += 180

        for i in range(1, M-1):
            for j in range(1, N-1):
                q = 255
                r = 255
                #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = gradient[i, j+1]
                    r = gradient[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = gradient[i+1, j-1]
                    r = gradient[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = gradient[i+1, j]
                    r = gradient[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = gradient[i-1, j-1]
                    r = gradient[i+1, j+1]
                if gradient[i,j] >= q and gradient[i,j] >= r:
                    Z[i,j] = gradient[i,j]
                else:
                    Z[i,j] = 0
        return Z
        
    def threshold(self,image, low, high):
        res = np.zeros(image.shape, dtype=np.float32)
        weak = 25
        strong = 255
        strong_i, strong_j = np.where(image >= high)
        zeros_i, zeros_j = np.where(image < low)
        weak_i, weak_j = np.where((image <= high) & (image >= low))
        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak
        return res, weak, strong


    def hysteresis(self,image, weak, strong=255):
        M, N = image.shape
        for i in range(1, M-1):
            for j in range(1, N-1):
                if image[i,j] == weak:
                    if ((image[i+1, j-1] == strong) or (image[i+1, j] == strong) or (image[i+1, j+1] == strong)
                        or (image[i, j-1] == strong) or (image[i, j+1] == strong)
                        or (image[i-1, j-1] == strong) or (image[i-1, j] == strong) or (image[i-1, j+1] == strong)):
                        image[i, j] = strong
                    else:
                        image[i, j] = 0
        return image
    
    def canny_detector(self):
        # Step1 : Noise Reduction
        smooth_img = self.convolve(self.gray_img,self.gaussian)

        # Step2 : Gradient Calculation
        gradient, theta = self.sobel_filters(smooth_img)

        # Step3 : Non-max suppression
        nms_img = self.non_max_suppression(gradient,theta)


        # Step4 : Double Threshold
        threshold_img, weak, strong = self.threshold(nms_img, self.lowthreshold*np.max(nms_img), self.highthreshold*np.max(nms_img))

        # # Step5 : Edge tracking by hysteresis
        canny_img = self.hysteresis(threshold_img, weak,strong)

        return canny_img
    
if __name__ == '__main__':
    
    # Read the image
    img = cv2.imread('edge-detectors/images/1.jpg')
    gray_img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
    # Ensure the values are in the 0-255 range and convert to uint8
    gray_img = np.clip(gray_img, 0, 255).astype(np.uint8)

    # Apply canny edge detector
    canny_detect = canny(gray_img)
    canny_image = canny_detect.canny_detector()
    
    cv2.imshow('Original image',gray_img)
    cv2.imshow('Canny edge detector',canny_image)
    
    cv2.waitKey(0)
   