# Edge Detection in Computer Vision

## Introduction 
Edge detection is the process of identifying sharp changes in image intensity to capture object boundaries. Edge detecion is the basis for object detection and image processing. 

## Methods for Edge Detection
- **Gradient-Based Methods**: Detecting edges by examining the maximum and minimum of the first derivative of the image. 
- **Second Derivative Methods**: Using the seconf derivative. eg:- Laplacian method

## Edge Detection Algorithms
- **Sobel Operator**
- **Canny Edge Detector**
- **Prewitt OPerator**
- **Laplacian of Gaussian (LoG)**
- **Roberts Cross Operator**


## Applications of Edge Detection
1. Image Segmentation
2. Object Detection
3. Feature Extraction



## Algorithms Implementation
1. Sobel Operator
- Sobel Operator is a widely used edge detection algorithm in image processing. It works by calculating the gradient of the image intensity at each pixel.
- An image can be considered as a two-dimensional function \(f(x,y) \), where the value of \( (x,y) \) represents the pixel intensity.
- **Gradient**: The gradient of \( f \) at any point is a 2D vector with components representing the derivatives in the horizontal and vertical directions   
   \[ \nabla f = \begin{bmatrix} G_x \\ G_y \end{bmatrix} = \begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{bmatrix} \]
   
- **Sobel Kernels**: Sobel operator uses two 3x3 convolution kernels to approximate the derivatives
   Horizontal changes (Gx):
  \[
  G_x = \begin{bmatrix}
  -1 & 0 & +1 \\
  -2 & 0 & +2 \\
  -1 & 0 & +1
  \end{bmatrix}
  \]

  Vertical changes (Gy):
  \[
  G_y = \begin{bmatrix}
  +1 & +2 & +1 \\
   0 &  0 &  0 \\
  -1 & -2 & -1
  \end{bmatrix}
  \]

### Convolution Process

1. **Convolution**: Applying the Sobel operator involves convolving each kernel with the image:

   \[ (f * g)(x, y) = \sum_{dx=-a}^{a} \sum_{dy=-b}^{b} f(x+dx, y+dy) \cdot g(dx, dy) \]

2. **Applying Kernels**: The Sobel kernels \( G_x \) and \( G_y \) are convolved with the image to produce horizontal and vertical derivative approximations.

### Gradient Magnitude and Direction

1. **Gradient Magnitude**: The edge strength at each pixel is given by the magnitude of the gradient:

   \[ \text{Magnitude} = \sqrt{G_x^2 + G_y^2} \]

2. **Gradient Direction**: The edge direction is calculated as:

   \[ \text{Direction} = \arctan\left(\frac{G_y}{G_x}\right) \]
