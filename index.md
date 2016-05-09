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

#### 1. Energy computation
 
An energy function is used to determine the importance of each pixel in the image. In my implementations, I defined the energy function has the sum of the vertical and horizontal gradients across all 3 channels of the image. <code>E(i,j) = E_r(i,j) + E_g(i,j) + E_b(i,j), where E_r(i,j)</code> is the sum of the horizontal and vertical gradients in the image's red channel. <code>E_r(i,j) = 0.5 * (abs(I(i, j-1) - I(i, j+1)) + abs(I(i+1, j) - I(i-1, j)))</code>. I ignore border pixels for simplicity (and to reduce branch divergence in my CUDA kernels). 

#### 2. Cumulative minimum energy computation (aka creating a "seam map")
 
Identifying the optimal (minimum energy) seam is done using dynamic programming to compute the cumulative minimal energy of each pixel. For pixels from the second to the last row, the cumulative minimal energy is <code>M(i, j) = E(i, j) + min(M(i-1, j-1), M(i-1, j), M(i-1, j+1))<code>. While for pixels in the first row, <code>M(1,j) = E(i, j)<code>. 

<img src="https://raw.githubusercontent.com/mopewaO/15418-project/gh-pages/images/min_seam.png" height="360px" style="float:center;"/>

#### 3. Backtracking for minimum seam identification
 
The pixel with the lowest M in the last row from step 2 represents the end of the vertical seam. In this step we backtrack from this pixel to the top of the image, the path of the backtrack representing the optimal seam.

#### 4. Minimum seam removal/insertion
 
The final step is then to remove the found identified pixels from the image. 

These 4 steps are repeated N times to reduce or enlarge the in either dimensions by n pixels. Repeating all the steps is very time consuming, especially for large images and/or removing many seams, thus real-time application is not possible. 

### Approach

My project focused on exploring different ways to parallelize the seam carving algorithm. I mainly focused on improving a parallel implementation of the energy computation and seam map computation functions (but every step is parallelized except for step 3, backtracking). 

#### Parallelizing Energy Computation: Naive Approach
For every pixel, its four neighbors' are needed to compute the vertical and horizontal gradients to get the energy value. Therefore, computing energy values is straight-forward to parallelize when doing the simple thing first. I divided the image into NxN tiles and mapped those directly to CUDA blocks. Then each thread corresponded to a pixel and each thread loaded the 12 values needed to compute that pixels energy (4 neighbors * 3 channels = 12). 

#### Parallelizing Energy Computation: Optimizations
In th above implementation, each thread carries out 12 global memory accesses even though these values can be shared among some threads. To reduce the large number of global memory accesses, I wrote another implementation that takes advantage of CUDA's shared memory. The image is still divided into NxN tiles but now the tiles are mapped to (N+2)x(N+2) CUDA blocks. Each thread first transfers the value of its assigned pixel to shared memory then I sync the threads to ensure all loads are complete before computing the pixel energies. (N+2)x(N+2) sized blocks are needed to also load pixels values adjacent to the tile. 

Additionally since we'll be dealing with very large images, I changed my data types to UInt16 to try to optimize for how much data we can fit into the processor's cache. 

#### Parallelizing Seam Map Computation
Step 2 is not as clearly parallelizable as step 1. For computing the values in the k-th row, all values in the (k-1)th row need to be computed beforehand. To ensure this, I have to repeatedly load the kernel for each row by putting it in a for loop that runs for the second row to the last row of the image. Although I decrease the number of global memory accesses by taking advantage of shared memory within a kernel, there is a big overhead attached to repeatedly launching a kernel for every row in the image. 

#### Optimizing
Although Step 2 limits our parallelization of Seam Carving somewhat, the bigger obstacle to achieving more speedup is having to remove seams from an Image one at a time. Can we get acceptable results if we only compute the energy Matrix and Seam Map once then remove minimum seams all at once? How many seams can we remove before the image starts becoming distorted. The steps for this new algorithm for seam carving are: 

1. Compute energy matrix 
2. Compute seam map
3. Find N smallest energy seams
4. Remove all N seams 

The big advantage here being that we are not looping N times to remove N seams one at a time. 

Unfortunately, due to struggles with CUDA Thrust and a bug I just couldn't track down, I was unable to implement the above steps to answer the questions I wanted to. 

### Results
Below are the results of the CUDA implementation of Seam Carving removing 200, 400, 600, 800, and 1000 seams from an image.

<table style="width:100%">
	<tr>
	<th>Original Image</th>
	<th>200 seams removed</th>
	</tr>
	<tr>
	<td width="50%" align="center"><img src="https://raw.githubusercontent.com/mopewaO/15418-project/gh-pages/images/tree.jpg" height="360px" style="float:center;"/></td>
	<td width="50%" align="center"><img src="https://raw.githubusercontent.com/mopewaO/15418-project/gh-pages/images/tree_200.jpg" height="360px" style="float:center;"/></td>
	</tr>
</table>
<br/><br/>

<table style="width:100%">
	<tr>
	<th>400 seams removede</th>
	<th>600 seams removed</th>
	</tr>
	<tr>
	<td width="50%" align="center"><img src="https://raw.githubusercontent.com/mopewaO/15418-project/gh-pages/images/tree_400.jpg" height="360px" style="float:center;"/></td>
	<td width="50%" align="center"><img src="https://raw.githubusercontent.com/mopewaO/15418-project/gh-pages/images/tree_600.jpg" height="360px" style="float:center;"/></td>
	</tr>
</table>
<br/><br/>

<table style="width:100%">
	<tr>
	<th>800 seams removede</th>
	<th>1000 seams removed</th>
	</tr>
	<tr>
	<td width="50%" align="center"><img src="https://raw.githubusercontent.com/mopewaO/15418-project/gh-pages/images/tree_800.jpg" height="360px" style="float:center;"/></td>
	<td width="50%" align="center"><img src="https://raw.githubusercontent.com/mopewaO/15418-project/gh-pages/images/tree_1000.jpg" height="360px" style="float:center;"/></td>
	</tr>
</table>
<br/><br/>

### References
1. Avidan S., Shamir A. Seam Carving for Content-Aware Image Resizing. ACM Trans. Graph. Vol 26., No. 3, 2007
2. Rubinstein M, Shamir A,Avidan S. Improved seam carving for video retargeting. ACM Trans Graph(SIGGRAPH), 2008, 27(3): 1-9
3. V. Vineet and P. J. Narayanan, "CUDA cuts: Fast graph cuts on the GPU," Computer Vision and Pattern Recognition Workshops, 2008. CVPRW '08. IEEE Computer Society Conference on, Anchorage, AK, 2008, pp. 1-8.
