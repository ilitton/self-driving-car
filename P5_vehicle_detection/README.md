# Vehicle Detection

### Demo:
<p align="center">
 <a href="https://drive.google.com/open?id=1Jy2bwvPXNz__BgIK4cY-T7PN2cuN7Nec"><img src="./output/project_video_out.gif" alt="Overview"></a>
 <br>Click for full video.
</p>

### Goals/Steps:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector
* Implement a sliding-window technique and use trained classifier to search for vehicles in images
* Create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles
* Estimate a bounding box for vehicles detected

### Repo Structure: 
* [`P5.ipynb`](P5.ipynb): solution and walkthrough (**TODO**: split the code into the following separate scripts):
 * [`feature_utils.py`](feature_utils.py): helper functions for feature extraction (HOG features, binned color features, color histogram features)
 * [`train.py`](train.py): script for splitting train and test data and training the model 
 * [`detection_utils.py`](detection_utils.py): helper functions for sliding window, heatmap, and car detection
 * [`hog.py`](hog.py): main script to process images with hog + SVM pipeline
* [`./test_images/`](./test_images/): images for testing pipline on single frames
* [`./output/`](./output/): any image or video created during pipeline development
* [`./data/`](./data/): pickled files of feature extraction and model metadata

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

  * The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

* Labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier
  * These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.

[//]: # (Image References)

[image1]: ./output/car_not_car_example.png "Data Examples"
[image2]: ./output/hog_example.png "HOG Example"
[image3]: ./output/test_pipeline.png "Pipeline Test"
[image4]: ./output/sliding_window.png "Sliding Window + Hog Supsampling"
[image5]: ./output/heatmap.png "Heatmap Mask"
[image6]: ./output/heatmap_label.png "Heatmap Label"
[image7]: ./output/bounding_boxes.png "Car Detection Bounding Boxes"

---

### 1. Histogram of Oriented Gradients (HOG)

Histogram of Oriented Gradients (HOG) is a feature descriptor that tries to extract information for object detection and throw away any information not useful for detection.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

A downside to HOG is the number of parameters to tune: 

 * `orient`: Number of orientation bins
 * `pix_per_cell`: Size of cell to be accumulated (in pixels)
 * `cell_per_block`: Number of cells in each block 
 * `hog_channel`: Which channel of the image to apply HOG to

I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). Here is an example using HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` on random images from each of the two classes to get a feel for what the HOG output looks like:

![alt text][image2]

I used a gridsearch for various feature parameter combinations and compared their performance with the same SVM classifier. Other parameters for the features include: 

 * `color_space`: Color space of output image (Ex: `RGB`, `HSV`, etc.)
 * `hist_bins`: Number of bins for the color histogram 
 
The final choice of feature parameters were chosen based on the model with the highest accuracy:

```
feature_param_dict = {'cell_per_block': 2,
  					  'color_space': 'YCrCb',
  					  'hist_bins': 16,
  					  'hist_feat': True,
  					  'hist_range': (0, 256),
  					  'hog_channel': 'ALL',
  					  'hog_feat': True,
  					  'orient': 9,
  					  'pix_per_cell': 16,
  					  'spatial_feat': True,
  					  'spatial_size': (16, 16)}
```

### 2. LinearSVM

I trained a linear SVM to classify whether or not a car is in the image with a gridsearch for the following parameters: `penalty` and `C`. The `C` parameter is the penalty parameter for the error term. 

First the HOG, binned color, and color histogram features are extracted from the car and not car images:

```
car_features = extract_features_from_files(car_images, feat_dict)
not_car_features = extract_features_from_files(not_car_images, feat_dict)
X = np.vstack((car_features, not_car_features)).astype(np.float64)
```

Then the features are standardized to bring all features to the same scale:

```
feature_scaler = StandardScaler().fit(X)
scaled_X = feature_scaler.transform(X)
```

Then the training happens:

```
# Create the classifier
svc = LinearSVC()

# Train the classifier
svc.fit(X_train, y_train)
```

The following model was chosen because it resulted in the highest accuracy (98.7%):
`svc = LinearSVC(c=0.005)`

Note: There was one large overall gridsearch with varying HOG parameters and SVM parameters, not two separate searches. 

### 3. Sliding Window Search

The sliding window search is combined with HOG sampling and is implemented by the function `detect_cars`. The function:

 1. Extracts and scale the features (HOG, binned color, color histogram) on a window 
 3. Subsample features according to a region within the window (think of this as a `subwindow`)
 3. Classifies whether or not a car is within that region
 4. Saves the window where the car is detected

The HOG feature extraction is combined with the sliding window approach to increase the speed of the algorithm. Calculating the features once and subsampling is much faster versus extracting the features individually on every single window. 

The search is restricted by the parameters `y_start` and `y_stop` which I restrict to the lower half of the image where the lanes are.

To optimize the performance of the classifier I tried training with different feature parameters, model parameters as well as window sizing. Ultimately I searched on 1.5 scales using the configuration listed in Section 1 above (YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector) which provided the following result:

![alt text][image3]

### 4. Heatmap

In order to add a filter for false positives and combine overlapping bounding boxes a heatmap was added with the following approach:

 1. I recorded the positions of positive detections in each frame of the video with `detect_cars`

 ![alt text][image4]

 2. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  

 ![alt text][image5]

 3. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  
 
 ![alt text][image6]
 
 4. Assuming each blob corresponded to a vehicle, I constructed bounding boxes to cover the area of each blob detected.

 ![alt text][image7]

---

### Video Implementation

Here's a clip of the pipeline's performance on a dashcam video:

<p align="center">
 <a href="https://drive.google.com/open?id=1Jy2bwvPXNz__BgIK4cY-T7PN2cuN7Nec"><img src="./output/project_video_out.gif" alt="Overview"></a>
 <br>Click for full video.
</p>

---

### Discussion

The HOG subsampling with SVM classification is dependent on parameter choice for feature extraction and training which makes the pipeline not very robust. With so many parameters to tune for feature extraction, training, and detection there's also a high chance of overfitting. A possible way to improve the pipeline's detection would be data augmentation with different conditions (Example: lighting).

To try and develop a more robust vehicle detection pipeline I could turn to deep learning, specifically the [You Only Look Once (YOLO)](https://pjreddie.com/darknet/yolo/) approach. Deep learning has become the state of the art approach for object detection and capture a better representation of images. The YOLO model has proven to be a fast and accurate object detector and has become the first next step I would like to do in order to improve the model performance. 

---
### Next Steps: YOLO

YOLO reframes the object detection problem as a regression problem where the model starts with image pixels and goes straight to bounding box coordinates. With YOLO, "you only look once at an image to predict where objects are present and where they are". YOLO has the advantage of seeing an entire image during training and detection versus the sliding window approach. 

The Tiny YOLO with Keras will have the following architecture: 

| Layer        | Depth           | Kernel Size | Stride Size  | Activation | Pooling
| :------------- |:-------------| :-----| :-- | :-- | :-- | :-- |
| Input (448x448x3)     |  |  |
| Convolutional | 16 | 3x3| 1x1| LeakyReLU| (2,2), `same`
| Convolutional | 32| 3x3| 1x1| LeakyReLU| (2,2), `same`
| Convolutional | 64| 3x3| 1x1| LeakyReLU| (2,2), `same`
| Convolutional | 128| 3x3| 1x1| LeakyReLU| (2,2), `same`
| Convolutional | 256| 3x3| 1x1| LeakyReLU| (2,2), `same`
| Convolutional | 512| 3x3| 1x1| LeakyReLU| (2,2), `same`
| Convolutional | 1024| 3x3| 1x1| LeakyReLU| (2,2), `same`
| Convolutional | 1024| 3x3| 1x1| LeakyReLU| (2,2), `same`
| Convolutional | 1024| 3x3| 1x1| LeakyReLU| (2,2), `same`
| Flatten|
| Fully Connected | 256 | 
| Fully Connected | 4096 | |  |LeakyReLU| 
| Fully Connected | 1470 | |  |
