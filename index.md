---
layout: default
title: CUDA Seam Carving
---
# Parallelizing Seam Carving on the GPU with CUDA, an Exploration

## Mopewa Ogundipe

### Summary
I explored new ways to parallelize seam carving on the GPU using CUDA.

### Background
Traditional image resizing techniques, such as image scaling, resampling, or cropping, are oblivious to the image content or not "content aware". Therefore changing an image's aspect ratio with these techniques causes significant distortions. Avidan and Shamir [1] proposed Seam Carving as a content-aware image resizing alternative. Seam Carving resizes the image by adding or removing "seams". A seam is a path of least importance through an image. It is defined as a 8-connected path of low "energy" pixels crossing the image (either vertically or horizontally), where only one pixel in a row or column (respectively) belongs to a seam. Seam Carving can be divided in 4 steps:

1. ###### Energy computation
 
An energy function is used to determine the importance of each pixel in the image. In my implementations, I defined the energy function has the sum of the vertical and horizontal gradients across all 3 channels of the image. <code>E(i,j) = E_r(i,j) + E_g(i,j) + E_b(i,j), where E_r(i,j)</code> is the sum of the horizontal and vertical gradients in the image's red channel. <code>E_r(i,j) = 0.5 * (abs(I(i, j-1) - I(i, j+1)) + abs(I(i+1, j) - I(i-1, j)))</code>.

2. ###### Cumulative minimum energy computation (aka creating a "seam map")
 
Identifying the optimal (minimum energy) seam is done using dynamic programming to compute the cumulative minimal energy of each pixel. For pixels from the second to the last row, the cumulative minimal energy is <code>M(i, j) = E(i, j) + min(M(i-1, j-1), M(i-1, j), M(i-1, j+1))<code>. While for pixels in the first row, <code>M(1,j) = E(i, j)<code>. 

3. ###### Backtracking for minimum seam identification
 
The pixel with the lowest M in the last row from step 2 represents the end of the vertical seam. In this step we backtrack from this pixel to the top of the image, the path of the backtrack representing the optimal seam.

4.  ###### Minimum seam removal/insertion
 
The final step is then to remove the found identified pixels from the image. 

These 4 steps are repeated N times to reduce or enlarge the in either dimensions by n pixels. Repeating all the steps is very time consuming, especially for large images and/or removing many seams, thus real-time application is not possible. 

### Approach

My project focused on exploring different ways to parallelize the seam carving algorithm. I mainly focused on improving a parallel implementation of the energy computation and seam map computation functions (but every step is parallelized except for step 3, backtracking). 

###### Parallelizing Energy Computation: Naive Approach
For every pixel, its four neighbors' are needed to compute the vertical and horizontal gradients to get the energy value. Therefore, computing energy values is straight-forward to parallelize when doing the simple thing first. I divided the image into NxN tiles and mapped those directly to CUDA blocks. Then each thread corresponded to a pixel and each thread loaded the 12 values needed to compute that pixels energy (4 neighbors * 3 channels = 12). 

###### Parallelizing Energy Computation: Optimizations



### Results

### References
1. Avidan S., Shamir A. Seam Carving for Content-Aware Image Resizing. ACM Trans. Graph. Vol 26., No. 3, 2007
2. Rubinstein M, Shamir A,Avidan S. Improved seam carving for video retargeting. ACM Trans Graph(SIGGRAPH), 2008, 27(3): 1-9
3. V. Vineet and P. J. Narayanan, "CUDA cuts: Fast graph cuts on the GPU," Computer Vision and Pattern Recognition Workshops, 2008. CVPRW '08. IEEE Computer Society Conference on, Anchorage, AK, 2008, pp. 1-8.
