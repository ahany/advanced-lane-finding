#**Advanced Lane Finding** 

This project is part of Udacity's [Self-Driving Car Engineer Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013). The goal of the project is to use computer vision to detect lane lines in given videos, calculate the lane curvature and the offset of the vehicle from lane center.

The steps of this project are the following:

 - Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
 - Apply a distortion correction to raw images.
 - Use color transforms, gradients, etc., to create a thresholded binary image.
 - Apply a perspective transform to rectify binary image ("birds-eye view").
 - Detect lane pixels and fit to find the lane boundary.
 - Determine the curvature of the lane and vehicle position with respect to center.
 - Warp the detected lane boundaries back onto the original image.
 - Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Below I go through the steps mentioned above in more details.

###**Camera Calibration**

The code for this step is contained in the file called `cam_calib.py`.  

For camera calibration, a set of chessboard images taken at different angles and distances is used. The idea is to map the coordinates of the corners in the chessboard 2D images which are called to `imgpoints` to the 3D coordinates (x, y, z) of the real undistorted chessboard corners called `objpoints`.

Here I am assuming the chessboard is fixed on the (x, y) plane at z = 0, such that the object points are the same for each calibration image. 

     # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
     objp = np.zeros((CORNERS_PER_ROW*CORNERS_PER_COLUMN, 3), np.float32)
     objp[:, :2] = np.mgrid[0:CORNERS_PER_ROW, 0:CORNERS_PER_COLUMN].T.reshape(-1, 2)

Next I prepared `imgpoints` by detecting the corners of the chessboard in the distorted calibration image using OpenCV's function`cv2.findChessboardCorners()`.  
A copy of `objp` every time I successfully detect all chessboard corners in a test image. 

    retval, corners = cv2.findChessboardCorners(gray, (CORNERS_PER_ROW, CORNERS_PER_COLUMN), None)
    if retval:
        imgpoints.append(corners)
        objpoints.append(objp)

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

Camera calibration matrix and distortion coefficients are pickled to be loaded when needed.

I applied this distortion correction to a test image using the `cv2.undistort()` function and obtained this result: 

![enter image description here](https://github.com/ahany/advanced-lane-finding/blob/master/output_images/test3_undist.jpg)

###**Pipeline (frame images)**

####**1. Distortion Correction**
The first step in the pipeline is to correct distortion. This is done in two steps:

 1. Loading the pickled calibration matrix and distortion coefficients using helper function `get_distortion_coeff()` in file `LaneDetector.py`

		mtx, dist = get_distortion_coeff() 
  
 2.  Applying OpenCV's function `cv2.undistort` to undistort each individual image using helper function `image_undistort()`in file `HelperFunctions.py`. 

	    image = image_undistort(frame, mtx, dist)

 
 Below is an example of an image before and after distortion correction.

Before:

![enter image description here](https://github.com/ahany/advanced-lane-finding/blob/master/output_images/test6.jpg)

After:

![enter image description here](https://github.com/ahany/advanced-lane-finding/blob/master/output_images/test6_undist.jpg)

####**2. Binary Thresholding**
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in function `image_mask()` at lines 72 through 97 in `HelperFunctions.py`).  I applied color extraction of white and yellow pixels along with absolute sobel thresholds. I found by experimentation that applying sobel thresholding on HLS channels yielded better results. I used L and S channels.

	masked = image_mask(image)

Here's an example of my output for this step:


![enter image description here](https://github.com/ahany/advanced-lane-finding/blob/master/output_images/test3_undist_masked.jpg)

####**3. Perspective Transform**

I implemented a class`PerspectiveTransformer` which allows to compute the transformation matrix only once and use it to apply transform or inverse transform on the fly

    class PerspectiveTransformer:
	    def __init__(self, src, dst):
		    self.src = src
		    self.dst = dst
		    self.Mat = cv2.getPerspectiveTransform(src, dst)
		    self.Mat_inv = cv2.getPerspectiveTransform(dst, src)
		  
		def transform(self, img):
			return cv2.warpPerspective(img, self.Mat, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

		def inverse_transform(self, img):
			return cv2.warpPerspective(img, self.Mat_inv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

I used the following **src** and **dst** points:
```
OFFSET = 117
SRC = np.array([[585, 460], [203, 720], [1111, 720], [702, 460]]).astype(np.float32)
DST = np.array([[203+OFFSET, 0], [203+OFFSET, 720], [1111-OFFSET, 720], [1111-OFFSET, 0]]).astype(np.float32)

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1111, 720     | 994, 720      |
| 702, 460      | 994, 0        |


To apply perspective transform, the **transfrom()** method is used:

    binary_warped = perspectivetransform.transform(masked)

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![enter image description here](https://github.com/ahany/advanced-lane-finding/blob/master/output_images/test_birds_view_before.jpg)

![enter image description here](https://github.com/ahany/advanced-lane-finding/blob/master/output_images/test_birds_view_after.jpg)

Here is another example of a pipeline image:

![enter image description here](https://github.com/ahany/advanced-lane-finding/blob/master/output_images/curved_birds_view_before_2.jpg)

![enter image description here](https://github.com/ahany/advanced-lane-finding/blob/master/output_images/curved_birds_view_after_2.jpg)

####**4. Lane-line pixels identification**

In this step, pixels of lane lines are detected and a 2nd order polynomial is fitted. This is done in a series of steps:

First, use a histogram to detect the peaks along all the columns in the lower half of the image to get starting points for the left and right lines. 

The helper function `find_histogram_peaks()` in the file `LaneDetector.py` was used.

    leftx_base, rightx_base = find_histogram_peaks(binary_warped)

![enter image description here](https://github.com/ahany/advanced-lane-finding/blob/master/output_images/histogram.png)

Then sliding windows are used to find lane pixels starting from the base of the lane lines detected by the histogram and following the lines up to the top of the image returning coefficients for the fitted polynomials for each line. 

This is implemented in the helper function `find_histogram_peaks()`in the file `LaneDetector.py`. Below is an image showing how the output looks like:

![enter image description here](https://github.com/ahany/advanced-lane-finding/blob/master/output_images/sliding_windows.jpg)

Then a second order polynomial was fit to each lane line using `numpy.polyfit()`. This is how the output should look like:

![enter image description here](https://github.com/ahany/advanced-lane-finding/blob/master/output_images/fitted_poly.png)

In order to add robustness to lane detection, the absolute difference between obtained polynomial coefficients and the coefficients from the previous iteration. The new coefficients are discarded if the difference exceeds a threshold. The threshold was decided by trying different numbers and selecting the most robust behavior

The outlier rejection mechanism is implemented at lines 210 through 217 in `LaneDetector.py` and show below:

    # outlier removal
    error_right = np.absolute(right_fit[0] - right_fit_prev[0])
    if error_right > 0.0005:
        right_fit = right_fit_prev

    error_left = np.absolute(left_fit[0] - left_fit_prev[0])
    if error_left > 0.0005:
        left_fit = left_fit_prev

To smooth out the lane lines, I added filtering over the last 10 frames (lines 219 through 229 in `LaneDetector.py`)

        # filtering over n_frames
    n_frames = 10
    if filtered_right_fit is None:
        filtered_right_fit = right_fit
    else:
        filtered_right_fit = (filtered_right_fit * (n_frames - 1) + right_fit) / n_frames

    if filtered_left_fit is None:
        filtered_left_fit = left_fit
    else:
        filtered_left_fit = (filtered_left_fit * (n_frames - 1) + left_fit) / n_frames


####**5. Calculation of radius of curvature for lane lines and vehicle position from center** 

Calculating the radius of curvature was done by scaling the x and y coordinates of the lane pixels and fitting a new polynomial to the real-world data. The new real-world polynomial coefficients are then used in the formula below to calculate the curvature.

![enter image description here](https://github.com/ahany/advanced-lane-finding/blob/master/output_images/Curvature_formula.png)

I did this using the helper function `find_lane_curvature()`  in `LaneDetector.py`

The position of the car with respect to the lane line center is calculated by first finding the x points for the left and right lane lines the bottom of the image and calculating the mean of these two points to get the lane center. Here we assume the lane center is the midpoint at the bottom of the image between the two detected lines.

Next the car position is calculated as the difference between the lane center and the center of the image. The position is multiplied by a scaling factor to convert from pixels to meters.

This was implemented in the helper function `find_car_offset()`  in `LaneDetector.py`

####**6**. **Plotting result image back down onto the road with lane area identified**

The final step is plotting the result back down onto the road and this was done using the helper functions `compute_intercepts()` and `draw_polygon()`in `LaneDetector.py`

The function  `draw_polygon()`applies inverse perspective transform to warp the image back to the original image space.

Below is an example of the result on a test image:

![enter image description here](https://github.com/ahany/advanced-lane-finding/blob/master/output_images/final_image.jpg)

As a final step, the lane curvature and vehicle offset are projected on the final image at lines 263 through 265 in `process_frame()` function in `LaneDetector.py`

The image below is an example of the final output from the pipeline:

![enter image description here](https://github.com/ahany/advanced-lane-finding/blob/master/output_images/sample_pipeline_output.jpg)

###**Pipeline (video)**

Here's a [link to my video result](./project_video.mp4)

###Reflection

I found two things to have the most significant impact on the performance of my pipeline. First was the image masking part. It was relatively easy after some parameters tuning to get good results on the first video. However, it is challenging to find parameters that can generalize well and work under different lighting conditions.

I also noticed that the source and destination points for the perspective transform heavily affect the performance. Again those points can be easily selected based on a few images from one specific video where the lanes are straight. However, the selected points that can perfectly fit the first video will not be that perfect for the challenge video.

I think the key to achieve more robustness and generalization would be for the algorithm to auto detect mask parameters as well as source and destination points for the perspective transform based on analyzing the input image.

