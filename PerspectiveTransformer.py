import cv2


# Class to apply perspective transform or inverse transform to an image
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
