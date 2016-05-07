---
layout: default
title: CUDA Seam Carving
---
### Summary

I explored new ways to parallelize seam carving on the GPU using CUDA.

### Background
Traditional image resizing techniques, such as image scaling, resampling, or cropping, are oblivious to the image content or not "content aware". Therefore changing an image's aspect ratio with these techniques causes significant distortions. Avidan and Shamir [1] proposed Seam Carving as a content-aware image resizing alternative. Seam Carving resizes the image by adding or removing "seams". A seam is a path of least importance through an image. It is defined as a 8-connected path of low "energy" pixels crossing the image (either vertically or horizontally), where only one pixel in a row or column (respectively) belongs to a seam. Seam Carving can be divided in 4 steps:

1. Energy computation
2. Cumulative minimum energy computation (aka creating a "seam map")
3. Backtracking for minimum seam identification
4. Minimum seam removal/insertion

These 4 steps are repeated n times to reduce or enlarge the in either dimensions by n pixels

Repeating all the steps is very time consuming, especially for large images and/or removing many seams, thus real-time application is not possible. 

### Approach

My project focused on exploring different ways to parallelize the seam carving algorithm
