import cv2
import matplotlib.image as mpimg
import numpy as np
import glob
import pickle

CORNERS_PER_ROW = 9
CORNERS_PER_COLUMN = 6

# Make a list of calibration images
images = glob.glob("./camera_cal/calibration*.jpg")

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((CORNERS_PER_ROW*CORNERS_PER_COLUMN, 3), np.float32)
objp[:, :2] = np.mgrid[0:CORNERS_PER_ROW, 0:CORNERS_PER_COLUMN].T.reshape(-1, 2)

# Step through the list and search for chessboard corners
for idx, image in enumerate(images):
    img = mpimg.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Find the chessboard corners
    retval, corners = cv2.findChessboardCorners(gray, (CORNERS_PER_ROW, CORNERS_PER_COLUMN), None)
    # Extract the image name
    tokens = image.split("/")
    img_name = tokens[-1]

    # If found, add object points, image points
    if retval:
        imgpoints.append(corners)
        objpoints.append(objp)
        cv2.drawChessboardCorners(img, (CORNERS_PER_ROW, CORNERS_PER_COLUMN), corners, retval)
        write_name = './camera_cal/' + img_name + '_corners_found' + '.jpg'
        cv2.imwrite(write_name, img)
    else:
        print('Corners not found for image {}'.format(img_name))


# load one image to get image size
img = mpimg.imread("./camera_cal/calibration1.jpg")
image_size = (img.shape[0], img.shape[1])
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

# checking undistortion on a test image
test_img = mpimg.imread("./camera_cal/calibration1.jpg")
output_image = cv2.undistort(test_img, mtx, dist, None, mtx)
cv2.imwrite('undist_test_img.jpg', output_image)

# Save the camera calibration result for later use
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open("./cam_calib_pickle.p", "wb"))

