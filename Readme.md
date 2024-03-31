# Assignment 1: Computer Vision(CSL7360)


## Harris Corner Detection
I successfully implemented the Harris Corner detection algorithm from scratch in Python with the use of numpy and scipy library. 

The key steps in my implementation are:
    • Converting the image to grayscale to compute gradient.
    • Computing Image Gradients, which can be configured in the “Scratch” class, as currently I have used Sobel Kernels, but one can also replace the self.d_x and self.d_y with Scharr, Prewitt or Roberts Cross operator.
    • Computing Harris Matrix Components, by this I mean the computation of Ixx, Iyy and Ixy for this I used scipy.ndimage.gaussian_filter, with a sigma of 1.
    • Computing Harris response, previous step’s components were used to calculate determinant and trace, which further gave the final reponse given by the formula , here I have chosen k as 0.05 but one can modify it while creating the object for “Scratch” class. 
    • Threshold to  identify Corners, By thresholding Harris reponse matrix we can get potential corners , and for my implementation I have set the thresholds as the fraction of maximum and minimum response values.
    • Visualise the corners.

For my implementation, the user can also see the intermediate images before corner estimation by choosing verbose as “True”, but if we only need the final corner highlighted image then user should keep verbose as “False”.

Overall, my implementation needs 3 parameters only, 
    1. k: the empirical constant
    2. THRESHOLD_CORNER: fraction of max response for corners
    3. THRESHOLD_EDGE: fraction of max response for edges

As seen in the image in ipynb, both implementations detect most of the same prominent corners, but there are some differences in the exact corner locations and the number of corners detected. This is likely due to parameter choices.
And mine is noticeable slower since it is not optimized. 

## Stereo 3D Reconstruction

In this question, I implemented a stereo 3D reconstruction algorithm to reconstruct a 3D point cloud from a pair of stereo images captured from a stereo camera setup. The provided inputs were:

    • Left stereo image (bikeL.png)
    • Right stereo image (bikeR.png)
    • Intrinsic camera matrices for left and right cameras (bike.txt)

Overall approach involved the following key steps:
    • Load the Stereo Images and Camera Parameters
        ◦ Camera parameters are as follows:
            ▪ baseline = 177.288
            ▪ f = 5299.313
            ▪ cx = 1263.818, cx_1 = 1438.004 and cy = 977.763
            ▪ Intrinsic matrices can be seen in bike.txt as the above parameters are derived from there it self.
    • Compute Disparity Map, by using the cv2.StereoBM_Create I computed the disparity map between the left and right images. 
    • Compute Depth Map, by using the formula , I computed the depth map.
    • Compute 3d Point cloud, by using the above given parameters I obtained the Projection Matrix Q and then using that with cv2.reporjectImageTo3D obtained the 3D point cloud in “reconstructed.ply” file. Then Utilised the Open3D library to visualize that PLY file

As seen in the point cloud, the 3D structure of the scene (bike, ground, background) is clearly reconstructed from the stereo pair of images.

## Epipolar geometry

In this question, I explored the concept of Epipolar geometry utilised  a fundamental matrix (F) to calculate epipolar lines, and visualize them in two images captured from a static scene.

Overall Approach involved the following steps:
    • Loading the images and fundamental matrix.
    • Convert to Grayscale for feature detection and matching.
    • Doing Feature Matching using FLANN and using Filter matches to extract the matched point coordinates in pts1 and pts2 array. 
    • Computing Epilines using the cv2.computeCorrespondingEpilines function
    • Visualising the images for image1 to image2 and image2 to image1 in a top down subplot manner are images are in landscape mode.

The output lists corresponding points between image 1 and image 2, and vice versa. Interestingly, all the corresponding points in image 2 have the same y-coordinate (0). This suggests that the epipolar line in image 2 might be horizontal (or very close to horizontal).

The output shows a one-to-one correspondence for most points in image 1. This is ideal, but in some cases, there might be multiple possible corresponding points on the epipolar line in the second image due to noise or inaccuracies, or some implementation inconsistencies.


### All the results can be seen in the corresponding notebooks.

# Contributors
- [Gaurav Sangwan](https://github.com/gauravsangwan)