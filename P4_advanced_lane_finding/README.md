# Advanced Lane Finding

### Demo:
<p align="center">
 <a href="https://drive.google.com/file/d/1zDsKfFVersG_-80hMvogda4VfWhGz3kI/view?usp=sharing"><img src="./output/project_output.gif" alt="Overview"></a>
 <br>Click for full video.
</p>

### Goals/Steps:
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  Some images in `output` were for testing pipeline on single frames.

### Repo Structure: 
* [`P4.ipynb`](P4.ipynb): walkthrough and solution
* [`./camera_cal/`](./camera_cal/): images for camera calibration
* [`calibration.p`](calibration.p): pickled calibration result
* [`./output/`](./output/): any image or video created during pipeline development

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

---

[//]: # (Image References)

[image0]: ./output/camera_calibrate_1.png "Camera Calibration"
[image1]: ./output/camera_calibrate_2.png "Test Undistort"
[image2]: ./output/undistorted_comparison.png "Undistorted Example"
[image3]: ./output/binary_threshold_comparison.png "Binary Example"
[image4]: ./output/warped.png "Warp Example"
[image5]: ./output/lane_detection.png "Fit Visual"
[image6]: ./output/lane_detection_2.png "Fit Visual 2"
[image7]: ./output/curvature_radius_formula.png "Radius of Curvature Formula"
[image8]: ./output/output_example.png "Output"
[video1]: ./output/project_ouput_bigger.gif "Video Output"

### 1. Camera Calibration

Camera calibration compute the transformation between 3D objects and 2D image points. Steps to compute the camera matrix and distortion coefficients: 

* Prepare "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world
 * Assume the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image
 * Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time successfully detect all chessboard corners in a test image are successfully detected
 * `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection
 
![Camera Calibration][image0]

**Figure 1.** Camera Calibration Example
 
### 2. Distortion Correction

Distortion correction is necessary to ensure the geometrical shape of objects is represented consistently no matter where they appear in the the image.
 
* Use the output `objpoints` and `imgpoints` and the function `cv2.calirbateCamera()` function to compute the camera calibration and distortion coefficients 

Applying the distrotion correction to a test chessboard image with the `cv2.undistort()` function resulted in the following image:
![Test Undistort][image1]
**Figure 2.** Distortion Corrected Calibration Example

Here's a side by side comparison of a dashcam image with and without the distortion correction applied:
![Distortion Example][image2]
**Figure 3.** Distortion Dashcam Example

Although it might not look undistorted, you can somewhat see the car's hood is changed at the bottom of the image 😅

### 3. Color and Gradient Threshold

**Sobel**: The heart of the Canny Algorithm. The Sobel Function takes the derivative of an image in the *x* and *y* direction. Taking the gradient in the:

* *x* direction emphasizes edges closer to the vertical
* *y* direction emphasizes edgers closer to the horizontal

I used a combination of color and gradient thresholds to generate a binary image.  Here's an example of my output for this step:

![Binary Example][image3]
**Figure 4.** Binary Threshold Example

### 4. Perspective Transformation

Perspective transformation changes an image such that we are effectively viewing an object from a different angle or direction. We want a bird's eye view of the lane from the dashcam images.

The code for my perspective transform includes a function called `warp()`, which  takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points. The warp maps 4 points in the source image to 4 points in the destination image.

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Warped Example][image4]
**Figure 5.** Perspective Transformation Example

### 5. Detect lane lines 

A sliding window approach was used to identify lane-line pixels and fit their positions with a 2nd order polynomial. 

![Fit Visual 1][image5]
![Fit Visual 2][image6]

**Figure 6.** Lane Line Detection Examples

### 6. Determine lane curvature and vehicle position

The radius of curvature of the lane is calculated using the coefficients returned from the 2nd order polynomial (1st order, 2nd order, bias) with the formula:

![Radius of Curvature Formula][image7]

Which can be implemented with the following code:

* Fit a second order poly and convert from pixels to meters

```
left_fit_curve = np.polyfit(left_y * ym_per_pix, left_x * xm_per_pix, 2)
right_fit_curve = np.polyfit(right_y * ym_per_pix, right_x * xm_per_pix, 2)
```

* Calculate curvature radii

```
left_curve_radius = ((1 + (2*left_fit_curve[0]*y_vals + left_fit_curve[1])**2)**1.5)\
                            / np.absolute(2*left_fit_curve[0])
right_curve_radius = ((1 + (2*right_fit_curve[0]*y_vals + right_fit_curve[1])**2)**1.5)\
                            / np.absolute(2*right_fit_curve[0])
```

Assuming the camera is mounted at the center of the vehicle, the vehicle position can be calculated as the difference between the center of the image (`car_position`) and the midpoint of the detected lanes (`lane_center_position`):

```
car_pos = warped.shape[1]/2
lane_center_pos = (right_fit + left_fit) /2
center_dist = (car_pos - lane_center_pos) * xm_per_pix
```
**Note**: The variable `xm_per_pix` is used to convert from pixels to meters. 

## Output Example

Here's an example image of the result plotted back down onto the road with the lanes identified:

![Output Example][image8]

---

### Video Implementation

Here's a clip of the pipeline's performance on a dashcam video:

<p align="center">
 <a href="https://drive.google.com/file/d/1zDsKfFVersG_-80hMvogda4VfWhGz3kI/view?usp=sharing"><img src="./output/project_output.gif" alt="Overview"></a>
 <br>Click for full video.
</p>

---

### Discussion

Some improvements could be to make the pipeline more robust. Currently, the pipeline is contingent on the initial steps of color and gradient thresholding. If the thresholding performs poorly to identify the lane lines then the following steps will most likely fail. Some thoughts on how to stregthen this part of the pipeline would be dynamic thresholding or using a CNN for the lane detection.