#!/usr/bin/env python3

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

class CustomSobel():
    ''' Custom  sobel operator algorithm '''
    def __init__(self,img):
        self.img = img
        # Define horizontal and vertical kernels
        self.Gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        self.Gy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        self.kernel_h, self.kernel_w = self.Gx.shape # dims of kernel
        self.img_h, self.img_w = self.img.shape
        

    def convolve2d(self,kernel):
        output = np.zeros((self.img_h-self.kernel_h+1,self.img_w-self.kernel_w+1))
        for y in range(self.img_h-self.kernel_h+1):
            for x in range(self.img_w-self.kernel_w+1):
                output[y,x] = np.sum(self.img[y:y+self.kernel_h, x:x+self.kernel_w]*kernel).astype(np.float32)

        return output
    
    def apply_sobel(self):
        sobel_x = self.convolve2d(self.Gx)
        sobel_y = self.convolve2d(self.Gy)

        grad_magnitude = np.sqrt(sobel_x**2+ sobel_y**2)
        grad_magnitude *= 255.0/grad_magnitude.max()
        grad_direction = np.arctan2(sobel_y, sobel_x) * (180 / np.pi)

        # Display the results
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1), plt.imshow(self.img, cmap='gray'), plt.title('Original')
        plt.subplot(2, 2, 2), plt.imshow(sobel_x, cmap='gray'), plt.title('Sobel X')
        plt.subplot(2, 2, 3), plt.imshow(sobel_y, cmap='gray'), plt.title('Sobel Y')
        plt.subplot(2, 2, 4), plt.imshow(grad_magnitude, cmap='gray'), plt.title('Gradient Magnitude')
        plt.show()


class cv2Sobel():
    '''Sobel Operator using cv2.sobel'''
    def __init__(self,img):
        self.img = img
        

    def apply_sobel(self):
        sobel_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=3)  
        sobel_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=3)

        grad_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        grad_direction = np.arctan2(sobel_y, sobel_x) * (180 / np.pi)


        # Display the results
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1), plt.imshow(self.img, cmap='gray'), plt.title('Original')
        plt.subplot(2, 2, 2), plt.imshow(sobel_x, cmap='gray'), plt.title('Sobel X')
        plt.subplot(2, 2, 3), plt.imshow(sobel_y, cmap='gray'), plt.title('Sobel Y')
        plt.subplot(2, 2, 4), plt.imshow(grad_magnitude, cmap='gray'), plt.title('Gradient Magnitude')
        plt.show()

        

if __name__ == '__main__':
    path_to_image = './images/1.jpg'
    img = Image.open(path_to_image)
    gray_img = np.array(img.convert('L'))

    c = CustomSobel(gray_img)
    c.apply_sobel()

    # C = cv2Sobel(gray_img)
    # C.apply_sobel()












