from PerspectiveTransformer import PerspectiveTransformer
from moviepy.editor import VideoFileClip
from HelperFunctions import *
import numpy as np
import cv2
import pickle


OFFSET = 117
# source and destination points for perspective transform
SRC = np.array([[585, 460], [203, 720], [1111, 720], [702, 460]]).astype(np.float32)
DST = np.array([[203+OFFSET, 0], [203+OFFSET, 720], [1111-OFFSET, 720], [1111-OFFSET, 0]]).astype(np.float32)


# conversions in x and y from pixels space to meters
ym_per_pix = 30 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

# previous polynomial coefificients
left_fit_prev = np.array([0, 0, 0], dtype='float')
right_fit_prev = np.array([0, 0, 0], dtype='float')
# used for filtering polynomial coefficients
filtered_left_fit = None
filtered_right_fit = None


def get_distortion_coeff():

    # load distortion coefficients from pickle file
    with open("cam_calib_pickle.p", "rb") as input_file:
        cam_calib = pickle.load(input_file)
    mtx = cam_calib["mtx"]
    dist = cam_calib["dist"]
    return mtx, dist


def find_histogram_peaks(binaryimage):

    # Take a histogram along all the columns in the lower half of the image
    histogram = np.sum(binaryimage[binaryimage.shape[0] / 2:, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    return leftx_base, rightx_base


def compute_intercepts(ymax, leftfit, rightfit):

    # Return evenly spaced numbers over a specified interval [0...719] which is the y-axis
    ploty = np.linspace(0, ymax - 1, ymax)

    # Get x points for all y points (compute intercepts)
    left_fitx = leftfit[0] * ploty ** 2 + leftfit[1] * ploty + leftfit[2]
    right_fitx = rightfit[0] * ploty ** 2 + rightfit[1] * ploty + rightfit[2]

    return left_fitx, right_fitx


def draw_polygon(image, left_fitx, right_fitx):

    # Return evenly spaced numbers over a specified interval [0...719] which is the y-axis
    ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])

    # Create an image to draw the lines on
    color_warp = np.zeros_like(image).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = perspectivetransform.inverse_transform(color_warp)
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    return result


def find_car_offset(leftfit, rightfit, xmax):

    # Get x points for the bottom of the image
    l_fitx = leftfit[0] * 720 ** 2 + leftfit[1] * 720 + leftfit[2]
    r_fitx = rightfit[0] * 720 ** 2 + rightfit[1] * 720 + rightfit[2]

    # lane center is the midpoint at the bottom of the image between the two detected lines
    camera_center = np.mean([l_fitx, r_fitx])
    # calculate the offset of the lane center from the center of the image and convert from pixels to meters
    center_diff = (camera_center - xmax/2) * xm_per_pix
    # check offset is right or left
    if center_diff <= 0:
        off_center_pos = 'right'
    else:
        off_center_pos = 'left'
    return center_diff, off_center_pos


def find_lane_curvature(ymax, leftfit, rightfit):
    # Fit new polynomials to x,y in world space
    ploty = np.linspace(0, ymax-1, num=ymax)  # to cover same y-range as image
    y_eval = np.max(ploty)

    # Convert polynomials back to points for refitting in world space
    leftx = leftfit[0] * ploty ** 2 + leftfit[1] * ploty + leftfit[2]
    rightx = rightfit[0] * ploty ** 2 + rightfit[1] * ploty + rightfit[2]

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

    # Calculate the new radii of curvature in meters
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    # Compute the mean of the two curvaters
    curvature = np.mean([left_curverad, right_curverad])  # (left_curverad + right_curverad)/2
    return curvature


def find_lane_pixels(binary_image, leftx_base, rightx_base, nwindows=9, margin=100, minpix=100):

    # apply a sliding window to find lane pixels starting from the base of the lane lines detected by the histogram
    # and following the lines up to the top of the image
    # Parameters:
    # leftxbase: starting point for the left lane line
    # rightxbase: starting point for the right lane line
    # nwindows: number of sliding windows
    # margin: width of the windows +/- margin
    # minpix: minimum number of pixels found to recenter window
    # Return: coefficients for the fitted polynomial for each lane line

    global left_fit_prev
    global right_fit_prev
    global filtered_right_fit
    global filtered_left_fit

    # Set height of windows
    window_height = np.int(binary_image.shape[0] / nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    out_img = np.dstack((binary_image, binary_image, binary_image)) * 255
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_image.shape[0] - (window + 1) * window_height
        win_y_high = binary_image.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows for visualization
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low)
                          & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low)
                           & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each to get coefficients (3 points).
    # If the lane pixels were not detected, use previous polynomial coefficients
    if (len(lefty) > 0) & (len(leftx) > 0):
        left_fit = np.polyfit(lefty, leftx, 2)
    else:
        left_fit = left_fit_prev

    if (len(righty) > 0) & (len(rightx) > 0):
        right_fit = np.polyfit(righty, rightx, 2)
    else:
        right_fit = right_fit_prev

    # outlier removal
    error_right = np.absolute(right_fit[0] - right_fit_prev[0])
    if error_right > 0.0005:
        right_fit = right_fit_prev

    error_left = np.absolute(left_fit[0] - left_fit_prev[0])
    if error_left > 0.0005:
        left_fit = left_fit_prev

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

    # update previous variables used above
    left_fit_prev = filtered_left_fit
    right_fit_prev = filtered_right_fit

    return filtered_left_fit, filtered_right_fit


def process_frame(frame):

        # undistort the image
        mtx, dist = get_distortion_coeff()
        image = image_undistort(frame, mtx, dist)
        # apply color mask and absolute sobel threshold
        masked = image_mask(image)
        # apply perspective transform
        binary_warped = perspectivetransform.transform(masked)
        # find histogram peaks along all the columns in the lower half of the image
        # to get starting points for the left and right lines
        leftx_base, rightx_base = find_histogram_peaks(binary_warped)
        # use sliding windows to follow the lane lines from bottom of image to top and return polynomial coefficients
        left_fit, right_fit = find_lane_pixels(binary_warped, leftx_base, rightx_base)
        # compute intercepts from the coefficients
        left_fitx, right_fitx = compute_intercepts(image.shape[0], left_fit, right_fit)
        # draw the polygon between lane lines on the frame image
        result = draw_polygon(image, left_fitx, right_fitx)

        # calculate the offset of the car on the road
        center_diff, off_center_pos = find_car_offset(left_fit, right_fit, image.shape[1])
        # calculate the radius of curvature for the lane lines
        curvature = find_lane_curvature(image.shape[0], left_fit, right_fit)

        # draw the text showing curvature and offset on the frame image
        cv2.putText(result, 'Radius of Curvature: '+str(round(curvature, 3))+'(m)', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, 'Vehicle is '+str(abs(round(center_diff, 3)))+'(m) '+ off_center_pos+' of center', (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return result

# select which video to process 0: project_video.mp4 1: challenge_video.mp4
VIDEO_SELECTION = 0

videos = ['./input_videos/project_video.mp4', './input_videos/challenge_video.mp4']
project_output = ['processed_project_video.mp4', 'processed_challenge_video.mp4']
project_output = project_output[VIDEO_SELECTION]
clip1 = VideoFileClip(videos[VIDEO_SELECTION])

# create an instance of the perspective transform class
# created here instead of in process_frame() to create only one instance
perspectivetransform = PerspectiveTransformer(SRC, DST)
# process the video frames
white_clip = clip1.fl_image(process_frame)
# save to output video file
white_clip.write_videofile(project_output, audio=False)

