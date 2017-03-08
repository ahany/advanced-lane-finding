import cv2
import numpy as np


def gaussian_blur(img, kernel=5):
    # Apply Gaussian filtering
    filtered = cv2.GaussianBlur(img, (kernel, kernel), 0)
    return filtered


def extract_yellow(img):
    # extract yellow pixels
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, 20, 20), (80, 255, 255))

    return mask


def extract_white(img):
    # extract white pixels
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, 0, 160), (255, 80, 255))
    return mask


def mag_thresh(img_ch, sobel_kernel=3, thresh=(0, 255)):

    # take the gradient in x and y separately
    sobelx = cv2.Sobel(img_ch, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img_ch, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # calculate the magnitude
    abs_sobelxy = np.sqrt(np.square(sobelx) + np.square(sobely))
    # scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobelxy / np.max(abs_sobelxy))
    # create a binary mask where mag thresholds are met
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1
    return mag_binary


def dir_threshold(img_ch, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # take the gradient in x and y separately
    sobelx = cv2.Sobel(img_ch, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img_ch, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # take the absoute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # calculate the direction of the gradient
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    # create a binary mask where direction thresholds are met
    dir_binary = np.zeros_like(grad_dir)
    dir_binary[(grad_dir > thresh[0]) & (grad_dir <= thresh[1])] = 1
    return dir_binary


def abs_sobel_thresh(img_ch, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(img_ch, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(img_ch, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a mask of 1's where the scaled gradient magnitude is > thresh_min and < thresh_max
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1
    return grad_binary


def image_mask(image):
    # convert to HLS color map
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    # extract l and s channels
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # apply sobel thresholds to l and s channels
    sobelx_lchannel = abs_sobel_thresh(l_channel, orient='x', sobel_kernel=5, thresh=(20, 255))
    sobely_lchannel = abs_sobel_thresh(l_channel, orient='y', sobel_kernel=5, thresh=(20, 255))
    sobelx_schannel = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=5, thresh=(20, 255))
    sobely_schannel = abs_sobel_thresh(s_channel, orient='y', sobel_kernel=5, thresh=(20, 255))
    # apply color map for yellow and white pixels
    mask_white = extract_white(image)
    mask_yellow = extract_yellow(image)
    color_mask = cv2.bitwise_or(mask_yellow, mask_white)

    # combine sobel thresholds
    sobell = np.copy(cv2.bitwise_or(sobelx_lchannel, sobely_lchannel))
    sobels = np.copy(cv2.bitwise_or(sobelx_schannel, sobely_schannel))
    sobel = cv2.bitwise_or(sobell, sobels)

    # combine sobel and color map thresholds
    combined = cv2.bitwise_and(color_mask, sobel)
    # apply gaussian filtering
    filtered = gaussian_blur(combined, 7)
    return filtered


def image_undistort(img, mtx, dist):
    # undistort images
    return cv2.undistort(img, mtx, dist, None, mtx)







