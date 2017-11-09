# Vehicle Detection Project
**Liang Zhang**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier, for example Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG.png
[image3]: ./output_images/sliding_windows.jpg
[image4_1]: ./output_images/test1_raw.png
[image4_2]: ./output_images/test2_raw.png
[image4_3]: ./output_images/test3_raw.png
[image5]: ./output_images/test4.png
[image6]: ./output_images/test5.png
[image7]: ./output_images/test6.png
[video1]: ./output_videos/project_video.mp4
[video2]: ./output_videos/test_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Contents
* vehicle_dection.ipynb: the main program
* output_images: the folder containing the output images
* output_videos: the folder containing the output videos
* writeup_report.md: the report

### Histogram of Oriented Gradients (HOG)

#### 1. Extract HOG features from the training images.

The code for getting HOG features is contained in the first code cell of the IPython notebook, vehicle_detection.ipynb.

In the third code cell, I started by reading in all the `vehicle` and `non-vehicle` images. 

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![Left side are the original images and the right side are the HOG featured images][image2]

#### 2. Choosing HOG parameters.

I tried various combinations of parameters, especially for color spaces. I had problems to use LUV and YUV color spaces, which will lead to some "NAN error". In rest of color spaces, I found YCrCb is the most stable one, and I settled on the following parameters.

Parameters | Value
-----------|------ 
color_space | YCrCb
orient | 12 
pix_per_cell | 8 
cell_per_block | 2 
hog_channel | ALL

#### 3. Train a linear SVM classifier using my selected HOG features and color features.

I trained a linear SVM using a combination of HOG feature, binned color features and color histogram features. For binned color features, I took spatial_size = (32, 32). For the color histogram features, I took hist_bins = 32. 

I normalized the features and took a random split of the data with split_rate = 0.2  before training.
### Sliding Window Search

#### 1. Choosing scales and other related parameters for sliding window search. 

Firstly I tried various scales from 0.5 to 2.5 on all test images, and I also played around with the starting and end points in y direction. Finally I used the parameters in the following table.

Scale | start point on y | end point on y
------| ---------------- | --------------
0.8 | 400 | 500
1.0 | 400 | 528
1.5 | 400 | 650
2   | 450 | 660

#### 2. Examples of the performance of the classifier on test images

Ultimately I searched on four scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![test1.jpg][image4_1]

![test2.jpg][image4_2]

![test3.jpg][image4_3]

### Video Implementation

#### 1. Final video output. 

![Link to the test video][video2]

![Link to the project video][video1]

#### 2.  Filters for false positives and using heatmap for combining overlapping bounding boxes.
After I use multiscale sliding search windows to dectect boxes, I used a line function acting as track boundary to filter out all the false positives on the opposite track. 

From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from three frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

**Here are results for three frames: original image, heatmap and with bounding boxes**

![Frame 1][image5]
![Frame 2][image6]
![Frame 3][image7]

Note that I used the hotboxes from prevous three frames to enrich the positive detected boxes, which is mainly used for smoothing the tracking boxes and also removing some false positive boxes. See code line 34-39, 55-57 in the pipeline() functions.

### Discussion

#### 1. Problems encountered and approaches to resolve them
* Problem1
overfitting when using block_norm = "L2-Hys" 
=> I found L1 norm works better
* Problem2
Hard to detect the cars far away appeared in smaller sizes 
=> using multiscale search winodws rather than only using one scale
* Problem3
"False" positive of cars driving on the opposite lanes. The good thing is that the classifier can detect cars also in the opposite direction, but the bad thing is that the classifier can not tell if the cars are driving in the opposite direction. The problem with using a threshold in heatmap is that it is hard to find a robust threshold. I tried various multiscale search setup with no luck.
=> Since the classifer was only trained with two labels, cars and not cars, I think the linear SVM classifier itself works perfectly as expected. Hence, I manually wrote a line function to simulate the lane boundary, which in the realistic case should be extracted from sensor data. With this line function, I can tell if the detected boxes are on the opposite lanes.

#### 2. Outlook
* A better way to smoothly tracking the car positions. There is still a bit oscillation of positions of tracked cars. It would be nicer to use more position information over a serie of old frames to smooth the car position for the next frame.
* Try other meachine learning classifiers, for example non-linear support vector machines,  decision trees, even deep neural networks, etc.

