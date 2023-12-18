#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
	https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Code starts here:

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import glob
from sklearn.cluster import KMeans

#dependencies
def rotate(image, angle):
    (row,col) = image.shape
    # center of the image
    cy = row/2
    cx = col/2
    center = (cx,cy)
    # Rotate matrix
    R_matrix = cv.getRotationMatrix2D(center,angle, scale=1)
    #rotate the image using warpAffine
    rotated_image = cv.warpAffine(image,R_matrix,image.shape[1::-1],flags = cv.INTER_LINEAR)
    return rotated_image

def half_disc(radius):
	hdisc = np.zeros([radius*2,radius*2])
	r = radius**2
	for i in range(radius):
		m = (i-radius)**2
		for j in range(2*radius):
			if m+(j-radius)**2 < r :
				hdisc[i,j] = 1
	return hdisc

def binary(image, bin):
# create binary images  
	binary_image = image*0 #2d array with zero
	row = image.shape[0]
	col = image.shape[1]
	for i in range(row):
		for j in range(col):
			if image[i,j] == bin:
				binary_image[i,j] = 1
			else:
				binary_image[i,j] = 0
	return binary_image

def gradient(maps, num_bins, mask1,mask2):
	maps = maps.astype(np.float64)
	s =12
	grad = np.zeros((maps.shape[0],maps.shape[1],s))
	for i in range(s):
		chi_square = np.zeros((maps.shape))  # image*0
		for j in range(1,num_bins):
			temp = binary(maps,j)
			g = cv.filter2D(temp, -1, mask1[i])
			h = cv.filter2D(temp, -1, mask2[i])
			chi_square = chi_square + ((g - h) ** 2) / (g + h + 0.0001)
		grad[:, :, i] = chi_square

	return grad


def GaussianKernel(n, sigma):
    # GAUSSIAN KERNEL IN 2D
    size = int((n-1)/2)
    # where sigma is std.deviation and n is the dimension of kerenel (n x n)
    p = np.asarray([[(-y* np.exp(-1 * (x ** 2 + y ** 2) / (2 * (sigma**2)))) for x in range(-size, size + 1)] for y in range(-size, size + 1)])
    kernel = p/(2*np.pi*(sigma**2)**2)
    return kernel

def Gaussian1d(sigma, mean, x, order):
    x =np.array(x) - mean  # mean is zero, center position
    #Gaussian Function , first derivative
    G = (1/np.sqrt(2*np.pi*(sigma**2)))*(np.exp((-1*(x**2))/(2*(sigma**2))))
    if order == 0:
        return G
    elif order == 1:
        G = -G*(x/(sigma**2))
        return G
    else:
        G = G*(((x**2) - (sigma**2))/((sigma**2)**2)) 
        return G

def Gaussian2d(k, sigma):
    # Second derivative of Gaussian Kernel
	size = int((k - 1) / 2)
	s = np.asarray([[x ** 2 + y ** 2 for x in range(-size, size + 1)] for y in range(-size, size + 1)])
	Gauss = (1 / np.sqrt(2 * np.pi * (sigma**2))) * np.exp(-s / (2 * (sigma**2)))
	return Gauss

def LOG2D(size,sigma):
    size = int((size-1)/2)
    s = np.asarray([[x ** 2 + y ** 2 for x in range(-size, size + 1)] for y in range(-size, size + 1)])
    p = (1 / np.sqrt(2 * np.pi * (sigma**2))) * np.exp(-s / (2 * (sigma**2)))
    Laplacian = p * (s - (sigma**2)) / ((sigma**2) ** 2)
    return Laplacian 

def makefilter(scale, phasex, phasey, Pts, size):
    Gx = Gaussian1d(3*scale,0,Pts[0, ...], phasex)
    Gy = Gaussian1d(scale, 0, Pts[1, ...], phasey)
    filter = Gx * Gy
    filter = np.reshape(filter,(size,size))
    return filter



'''Filter banks'''

def Oriented_DOG():
    #DoG is a spatial band-pass filter that attenuates frequencies in the original grayscale image that are far from the band center.
    # Two Gaussian kernels with different standatrd deviation, 2 Scales
    sigma = [1,2]
    num_orientations = 16
    orientation = np.linspace(0,360,num_orientations)   #different angles
    plt.figure(figsize = (20,8)) 
    plt.suptitle("OrientedDoG")
    Fil = []
    
    # copy 16x2 different Gaussian filters
    for i in range(0,len(sigma)):
        kernel =  GaussianKernel(7,sigma[i])  # Gaussian Kernel of 7x7
        for j in range(0, num_orientations):
            filters = rotate(kernel, orientation[j])
            Fil.append(filters)
            plt.subplot(3*len(sigma), num_orientations, num_orientations * (i) + j + 1)
            plt.axis('off')
            plt.imshow(Fil[num_orientations * (i) + j], cmap='gray')

    plt.show()
    return Fil


def LMS():
	sz = 57 
	scales = np.sqrt(2) ** np.array([0, 1, 2])  # sigma = 1, root(2), 2 , 2 root(2)
	num_Orientation = 6
	num_Filters = 48
	Filters = np.zeros([sz, sz, num_Filters])  # 57X57 48 filters
	hK = (sz - 1) / 2

	x = [np.arange(-hK, hK + 1)]
	y = [np.arange(-hK, hK + 1)]
	[x, y] = np.meshgrid(x, y)
	ref = [x.flatten(), y.flatten()]
	ref = np.array(ref)

	filter_count = 0

	for scale in range(len(scales)):
		for orient in range(num_Orientation):
			angle = (np.pi * orient) / num_Orientation
			cosine_ = np.cos(angle)
			sin_ = np.sin(angle)
			rotPts = [[cosine_, -sin_], [sin_, cosine_]]
			rotPts = np.array(rotPts)
			rotPts = np.dot(rotPts, ref)
			# print(rotPts)
			Filters[:, :, filter_count] = makefilter(scales[scale], 0, 1, rotPts, sz)
			Filters[:, :, filter_count + 18] = makefilter(scales[scale], 0, 2, rotPts, sz)
			filter_count += 1

	filter_count = 36   # 12 more filters, current count = 36
	scales = np.sqrt(2) ** np.array([1, 2, 3, 4])

	for i in range(len(scales)):
		Filters[:, :, filter_count] = Gaussian2d(sz, scales[i])
		filter_count += 1

	for i in range(len(scales)):
		Filters[:, :, filter_count] = LOG2D(sz, scales[i])
		filter_count += 1

	for i in range(len(scales)):
		Filters[:, :, filter_count] = LOG2D(sz, 3 * scales[i])
		filter_count += 1

	plt.figure(figsize=(20,10))
	for i in range(0, 48):
		plt.subplot(6, 8, i + 1)
		plt.axis('off')
		plt.imshow(Filters[:, :, i], cmap='gray')
		plt.suptitle("LMS")
	plt.show()
	return Filters

def LML():
	sz = 57  # 57X57 48 filters
	scales = np.sqrt(2) ** np.array([1, 2, 3])  
	num_Orientation = 6
	num_Filters = 48
	Filters = np.zeros([sz, sz, num_Filters])
	hK = (sz - 1) / 2

	x = [np.arange(-hK, hK + 1)]
	y = [np.arange(-hK, hK + 1)]
	[x, y] = np.meshgrid(x, y)
	ref = [x.flatten(), y.flatten()]
	ref = np.array(ref)

	filter_count = 0

	for scale in range(len(scales)):
		for orient in range(num_Orientation):
			angle = (np.pi * orient) / num_Orientation
			cosine_ = np.cos(angle)
			sin_ = np.sin(angle)
			rotPts = [[cosine_, -sin_], [sin_, cosine_]]
			rotPts = np.array(rotPts)
			rotPts = np.dot(rotPts, ref)
			Filters[:, :, filter_count] = makefilter(scales[scale], 0, 1, rotPts, sz)
			Filters[:, :, filter_count + 18] = makefilter(scales[scale], 0, 2, rotPts, sz)
			filter_count += 1

	filter_count = 36   # 12 more filters, curren t count = 36
	scales = np.sqrt(2) ** np.array([1, 2, 3, 4])

	for i in range(len(scales)):
		Filters[:, :, filter_count] = Gaussian2d(sz, scales[i])
		filter_count += 1

	for i in range(len(scales)):
		Filters[:, :, filter_count] = LOG2D(sz, scales[i])
		filter_count += 1

	for i in range(len(scales)):
		Filters[:, :, filter_count] = LOG2D(sz, 3 * scales[i])
		filter_count += 1

	plt.figure(figsize=(20,10))
	for i in range(0, 48):
		plt.subplot(6, 8, i + 1)
		plt.axis('off')
		plt.imshow(Filters[:, :, i], cmap='gray')
		plt.suptitle("LML")
	plt.show()

	return Filters



def gabor(sigma,theta, Lambda, psi, gamma):
    # A linear filter used for texture analysis
    Gabor = []
    num_filters =15
    for i in sigma:
        sigma_x = i
        sigma_y = float(i)/gamma

        # Boundary 
        std = 3  # Number of standard deviation sigma
        x_max = np.ceil(max(1,max(abs(std * sigma_x * np.cos(theta)), abs(std * sigma_y * np.sin(theta)))))
        y_max = np.ceil(max(1,max(abs(std * sigma_x * np.sin(theta)), abs(std * sigma_y * np.cos(theta)))))
        x_min = -x_max
        y_min = -y_max
        (y, x) = np.meshgrid(np.arange(y_min, y_max + 1), np.arange(x_min, x_max + 1))
        x_prime = x*np.cos(theta)+y*np.sin(theta)
        y_prime = -x *np.sin(theta)+y*np.cos(theta)
        Gb = np.exp(-1*((x_prime**2/(2*(sigma_x**2)))+(y_prime**2/(2*(sigma_y**2)))))*np.cos(((2*np.pi*x_prime)/Lambda)+psi)
        plt.figure('Gabor filters')
        angle  = np.linspace(0,360, num_filters)
        for i in range(num_filters):
            filter = rotate(Gb,angle[i])
            Gabor.append(filter)
        for i in range(len(Gabor)):
            plt.subplot(int(len(Gabor)/5),5,i+1)
            plt.axis('off')
            plt.imshow(Gabor[i], cmap = 'gray')
			# plt.suptitle('Gabor filters')
    plt.show()	
    return Gabor

def main():

	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""
	DoG  = Oriented_DOG()
	
	# """
	# Generate Leung-Malik Filter Bank: (LM)
	# Display all the filters in this filter bank and save image as LM.png,
	# use command "cv2.imwrite(...)"
	# """
	# L = LML()
	# l = LMS()

	# """
	# Generate Gabor Filter Bank: (Gabor)
	# Display all the filters in this filter bank and save image as Gabor.png,
	# use command "cv2.imwrite(...)"
	# """
	# Gabor = gabor([9,13],np.pi/6,7,0.5,1)

	# """
	# Generate Half-disk masks
	# Display all the Half-disk masks and save image as HDMasks.png,
	# use command "cv2.imwrite(...)"
	# """
	# orientations = np.arange(0,360, int(360/8))
	# scales = np.asarray([5,7,10])
	# mask1 = list()
	# mask2 = list()
	# size = scales.size
	# num_orientation =  orientations.size

	# for i in range(size):
	# 	hdisc = half_disc(scales[i])
	# 	for j in range(num_orientation):
	# 		msk1 = (rotate(hdisc,orientations[j]))
	# 		mask1.append(msk1)
	# 		msk2 = rotate(msk1, 180)
	# 		mask2.append(msk2)
	# 		plt.subplot(2*size, num_orientation,  ((2*num_orientation*i) + j + 1))
	# 		plt.axis('off')
	# 		plt.imshow(msk1, cmap='gray')
	# 		plt.subplot(2*size, num_orientation,  ((2*num_orientation*i) + j + 1))
	# 		plt.axis('off')
	# 		plt.imshow(msk2, cmap='gray')
	# plt.show()
            
    # # create the filter bank
    # # Append all the filters to create a filterbank
	
	

	# filter_bank = list()
	# for i in range(len(DoG)):
	# 	filter_bank.append(DoG[i])
	# for i in range(48):
	# 	filter_bank.append(L[:, :, i])
	# for i in range(48):
	# 	filter_bank.append(l[:, :, i])
	# for i in range(len(Gabor)):
	# 	filter_bank.append(Gabor[i])

	
	# os.chdir("../BSDS500/Images")

	# Images = []
	# for image in sorted(glob.glob("*.jpg")):
	# 	img = cv.imread(image)
	# 	Images.append(img)

	# Img_No = 1    # choose any image from the collection
	# plt.imshow(cv.cvtColor(Images[Img_No], cv.COLOR_BGR2RGB))
	# plt.show()
	# os.chdir("../../Code")

	# Img = cv.cvtColor(Images[Img_No], cv.COLOR_BGR2GRAY)
	# Imagex = Images[Img_No]

	# """
	# Generate Texton Map
	# Filter image using oriented gaussian filter bank
	# """
    # #Filter the images using the 'filetr_bank'
	# Img_fil  = np.zeros((Img.size, len(filter_bank)))
	# for i in range(len(filter_bank)):
	# 	temp = cv.filter2D(Img,-1, filter_bank[i]) # filtering image 'Img' using cv2.filter2D
	# 	temp = temp.reshape(1,Img.size)
	# 	Img_fil[:,i] = temp  # image filtered throught filters in the filter bank
    
	# # print(Img_fil.shape) 


	# """
	# Generate texture ID's using K-means clustering
	# Display texton map and save image as TextonMap_ImageName.png,
	# use command "cv2.imwrite('...)"
	# """
	# #KMeans clustering
	# num_clusters = 64  # Number of Clusters
	# # n_init, number of timesthe KMeans algorithm run with different centroid seeds
	# texton = KMeans(num_clusters,n_init = 4)
	# texton.fit(Img_fil)  # fittiing the filtered image to identify clusters and generate IDs
	# ID = texton.labels_  # labels of each point , ID
	# texton_map = np.reshape(ID,(Img.shape)) #reshape to the shape of the image
	# #plot map
	# plt.imshow(texton_map)
	# plt.title("Texton Map")
	# plt.axis('off')
	# plt.show()

	# """
	# Generate Texton Gradient (Tg)
	# Perform Chi-square calculation on Texton Map
	# Display Tg and save image as Tg_ImageName.png,
	# use command "cv2.imwrite(...)"
	# """
	# #Chi square Calculation on Texton Map
	# Tg = gradient(texton_map, 64, msk1, msk2)
	# tex_gradient_mean = np.mean(Tg, axis=2)
	# plt.imshow(tex_gradient_mean)
	# plt.title("Texton Gradient")
	# plt.axis('off')
	# plt.show()

	# """
	# Generate Brightness Map
	# Perform brightness binning 
	# """
	# (x,y) = Img.shape[0],Img.shape[1]
	# bin = Img.reshape(x*y,1)
	# KMeans_brightness = KMeans(n_clusters =16, random_state =4)
	# KMeans_brightness.fit(bin)
	# labels = KMeans_brightness.labels_
	# brightmap = np.reshape(labels,(x,y))
	# min = np.min(brightmap)  # minimum value
	# max = np.max(brightmap)  # maximum value
	# max_diff = max - min
	# brightness_map = 255*(brightmap - min)/max_diff

	# plt.imshow(brightness_map, cmap='gray')
	# plt.title('Brightness map')
	# plt.axis('off')
	# plt.show()
	
	# """
	# Generate Brightness Gradient (Bg)
	# Perform Chi-square calculation on Brightness Map
	# Display Bg and save image as Bg_ImageName.png,
	# use command "cv2.imwrite(...)"
	# """
	# # mean of brighness gradient for allthe pixels
	# brightness_gmean = np.mean(gradient(brightmap, 16, msk1, msk2), axis=2)
	# plt.imshow(brightness_gmean)
	# plt.title("Brightness Gradient")
	# plt.axis('off')
	# plt.show()

	# """
	# Generate Color Map
	# Perform color binning or clustering
	# """
	# # color map
	# x = Imagex.shape[0]
	# y = Imagex.shape[1]
	# bin = Imagex.reshape(x*y,3)  # color image
	# KMeans_color = KMeans(n_clusters = 16, random_state =4)
	# KMeans_color.fit(bin)
	# labels = KMeans_color.labels_
	# colormap = np.reshape(labels,(x,y))
	# plt.imshow(colormap)
	# plt.title("ColorMap")
	# plt.axis('off')
	# plt.show()
	# """
	# Generate Color Gradient (Cg)
	# Perform Chi-square calculation on Color Map
	# Display Cg and save image as Cg_ImageName.png,
	# use command "cv2.imwrite(...)"
	# """
	# color_gmean = np.mean(gradient(colormap, 16, msk1, msk2), axis=2)
	# plt.imshow(color_gmean)
	# plt.title("Color Gradient")
	# plt.axis('off')
	# plt.show()

	# """
	# Read Sobel Baseline
	# use command "cv2.imread(...)"
	# """
	# os.chdir("../BSDS500/SobelBaseline")
	# Images = []
	# for image in sorted(glob.glob("*.png")):
	# 	img = cv.imread(image)
	# 	Images.append(img)
	# img_sob = cv.cvtColor(Images[Img_No], cv.COLOR_BGR2RGB)
	# plt.imshow(img_sob)
	# plt.title("sobel output")
	# plt.axis('off')
	# plt.show()
	# """
	# Read Canny Baseline
	# use command "cv2.imread(...)"
	# """
	# os.chdir("../CannyBaseline")
	# Images = []
	# for image in sorted(glob.glob("*.png")):
	# 	img = cv.imread(image)
	# 	Images.append(img)
	# img_can = cv.cvtColor(Images[Img_No], cv.COLOR_BGR2RGB)
	# plt.imshow(img_can)
	# plt.title("Canny output")
	# plt.axis('off')
	# plt.show()

	# """
	# Combine responses to get pb-lite output
	# Display PbLite and save image as PbLite_ImageName.png
	# use command "cv2.imwrite(...)"
	# """
	# # PB-Lite output
	# # convert image to grayscale 
	# sobel_image = cv.cvtColor(img_sob, cv.COLOR_BGR2GRAY)
	# canny_image = cv.cvtColor(img_can, cv.COLOR_BGR2GRAY)
	# # Averaging all outputs (Tg,Bg,Cg)
	# Average = (tex_gradient_mean+brightness_gmean+color_gmean)/3
	# # comparing with base line
	# pb = Average * (0.5*sobel_image+ 0.5*canny_image)  # W1, w2 is taken as 0.5 , w1+w2 =1
    
	# # show the final output
	# plt.imshow(pb, cmap="gray")
	# plt.title("pblite-output")
	# plt.axis('off')
	# plt.show()
    
if __name__ == '__main__':
    main()
 


