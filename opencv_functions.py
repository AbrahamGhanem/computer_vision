import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

image_path = os.path.join(os.getcwd(), "result.jpg")
image = cv2.imread(image_path)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

resize_into_smaller_image = cv2.resize(image, (500, 500), cv2.INTER_LINEAR)

rows, cols = image.shape[:2]  # (col/2,rows/2) is the center of rotation for the image = M
M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)  # center, angle, scale
dst_1 = cv2.warpAffine(image, M, (cols, rows))  # img, trans-mat, destination dim

# shifting the image 100 pixels in both dimensions
M = np.float32([[1, 0, -100], [0, 1, -100]])
dst = cv2.warpAffine(image, M, (cols, rows))

ret, thresh_binary = cv2.threshold(gray_image, 127, 255,cv2.THRESH_BINARY)
ret, thresh_binary_inv = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh_trunc = cv2.threshold(gray_image, 127, 255, cv2.THRESH_TRUNC)
ret, thresh_tozero = cv2.threshold(gray_image, 127, 255, cv2.THRESH_TOZERO)
ret, thresh_tozero_inv = cv2.threshold(gray_image, 127, 255, cv2.THRESH_TOZERO_INV)


thresh_mean = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
thresh_gaussian = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


edges = cv2.Canny(image, 100, 200)


def image_filtering(image):
    # get a one dimensional Gaussian Kernel
    gaussian_kernel_x = cv2.getGaussianKernel(5, 1)
    gaussian_kernel_y = cv2.getGaussianKernel(5, 1)
    # converting to two dimensional kernel using matrix multiplication
    gaussian_kernel = gaussian_kernel_x * gaussian_kernel_y.T
    # you can also use cv2.GaussianBLurring(image,(shape of kernel),standard deviation) instead of cv2.filter2D
    filtered_image = cv2.filter2D(image, -1, gaussian_kernel)
    return filtered_image


# Scale Invariant Feature Transform (SIFT) - for: image stitching, object detection, etc.
def scale_invariant_feature_transform(gray_image):
    # create sift object
    sift = cv2.xfeatures2d_SIFT.create()  # only works on python < 3.4.2.
    # calculate keypoints and their orientation
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    # plot keypoints on the image
    with_keypoints = cv2.drawKeypoints(gray_image, keypoints)
    return with_keypoints


# Enhanced version of SIFT
def speededup_robust_features(gray_image):
    # instantiate surf object
    surf = cv2.xfeatures2d.SURF_create(400)
    # calculate keypoints and their orientation
    keypoints, descriptors = surf.detectAndCompute(gray_image, None)
    with_keypoints = cv2.drawKeypoints(gray_image, keypoints)
    return with_keypoints


# Feature Matching (using the features extracted from different images using SIFT or SURF)
def feature_matcher(image1, image2):
    sift = cv2.xfeatures2d_SIFT.create()
    # finding out the keypoints and their descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
    # matching the descriptors from both the images
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # selecting only the good features
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])
    image3 = cv2.drawMatchesKnn(image1, keypoints1, image2, keypoints2, good_matches, flags=2)
    return image3


cv2.imshow("image", feature_matcher(image, dst_1))
cv2.waitKey(0)