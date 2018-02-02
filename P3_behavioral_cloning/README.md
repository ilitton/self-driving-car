# Behaviorial Cloning Project

### Demo:
<p align="center">
 <a href="https://drive.google.com/open?id=1jpOyY3RY47zJsvpRgu6AWWXQKDUR4aD_"><img src="./writeup/run1_full_screen.gif" alt="Overview"></a>
 <br>Click for full video.
</p>


### Goals/Steps:

* Use the simulator (provided by Udacity) to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images to clone driving behavior
* Train and validate the model with a training and validation set
* Test that the model successfully drives around a track without leaving the road

### Repo Structure:
* [`model.py`](model.py): script used to create and train the model
* [`drive.py`](drive.py): script to drive the car in the simulator
* [`model.h5`](model.h5): a trained Keras model
* [`video.mp4`](video.mp4): a video of the vehicle driving autonomously in the simulator
* [`preprocess.py`](preprocess.py): script with preprocessing functions
* [`./data/`](./data/): directory with log containing steering wheel angles, image files names, and more
* [`./model/`](./model/): directory with model weights and metadata
* [`./writeup/`](./writeup/): directory with writeup images, videos, and gifs

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

[//]: # (Image References)

[image1]: ./writeup/center.png "Center"
[image2]: ./writeup/left.png "Left"
[image3]: ./writeup/right.png "Right"
[image4]: ./writeup/flipped_center.png "Flipped Image"
[image5]: ./writeup/nvidia_model.png "NVIDIA Architecture"

---

## Model Architecture

#### 1. Solution design approach

My convolutional neural network consists of 9 layers with 5 convolutional layers and 3 fully connected layers inspired by NVIDIA's CNN in [*End to End Learning for Self-Driving Cars*] (https://arxiv.org/pdf/1604.07316.pdf). The architecture is illustrated below:

![NVIDIA Architecture][image5]
**Figure 1.** NVIDIA Architecture

The only difference from Figure 1 above is I added cropping and resizing layers, which are not adjusted in the learning process. 
The model includes ELU layers to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer. The ELU activation was chosen based on Comma.ai's steering model [here](https://github.com/commaai/research/blob/master/train_steering_model.py).

The overall strategy for deriving a model architecture was to use NVIDIA's CNN model as a baseline and build upon it.

My first step was to use a CNN model similar to NVIDIA's model and see its performance the track. I thought this model might be appropriate because it's already demonstrated its success for this specific usecase. 

In order to gauge how well the model was working, I split my image and steering angle data into a training (80%) and validation (20%) set. The initial model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

#### 2. Attempts to reduce overfitting in the model

The model contains L2 regularization for every layer with a penalization of 0.001 in order to reduce overfitting. 

The model was trained and validated on different data sets by randomly splitting and shuffling the data to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually.

#### 4. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following design:

| Layer        | Depth           | Kernel Size | Stride Size  | Activation | Regularization|
| :------------|:----------------|:------------|:-------------|:-----------|:--------------|
| Input (160x320x3) |
| `Lambda`: Normalization |
| `Cropping2D` ((75,25), (0,0)) |
| `Lambda`: Resize (66x200x3) |
| Convolutional | 24| 5x5| 2x2| ELU| `l2`(0.000001) 
| Convolutional | 36| 5x5| 2x2| ELU| `l2`(0.000001) 
| Convolutional | 48| 5x5| 2x2| ELU| `l2`(0.000001) 
| Convolutional | 64| 3x3| 1x1| ELU| `l2`(0.000001) 
| Convolutional | 64| 3x3| 1x1| ELU| `l2`(0.000001) 
| `Flatten` |
| Fully Connected | 100 | |  |ELU| `l2`(0.000001)
| Fully Connected | 50 | |  |ELU| `l2`(0.000001) 
| Fully Connected | 10 | |  |ELU| `l2`(0.000001)
| Fully Connected | 1 |

The final step was to run the simulator to see how well the car was driving around track one. At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 5. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For more details about how I created the training data, please see the next section. 

## Training Strategy

#### 1. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Center Angle][image1]

**Figure 2.** Center Angle Example

Here are examples of center lane driving from the left angle:

![Left Angle][image2]

**Figure 3.** Left Angle Example

And the right angle:

![Right Angle][image3]

**Figure 4.** Right Angle Example

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back on track after veering off the track. 

Then I repeated this process on track two in order to get more data points. I also recorded two laps on both tracks driving the track from the opposite direction to get more training data.

To augment the data set, I also flipped images and angles. For example, here is the center angle example (**Figure 1**) and how it looks flipped:

![Center Angle][image1]

**Figure 1.** Center Angle Example

![Flipped Example][image4]

**Figure 5.** Flipped Example

After the collection process, I had 29,691 number of data points and 89,073 images. I then preprocessed this data by the following steps:

 1. Normalization: `Lambda(lambda x: x/127.5 - 1., input_shape=(160,320,3))`
 2. Cropping: `Cropping2D(cropping=((75,25),(0,0)))`
 3. Resize: `tf.image.resize_images(img, (66, 200))`

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the validation error plateauing after the 5th epoch. I used an adam optimizer so that manually training the learning rate wasn't necessary.

## Output Example

### Track 1

Here's an example of the model's performance on track 1 from the full screen of the simulator:

<p align="center">
 <a href="https://drive.google.com/open?id=1jpOyY3RY47zJsvpRgu6AWWXQKDUR4aD_"><img src="./writeup/run1_full_screen.gif" alt="Overview"></a>
 <br>Click for full video.
</p>

This is an example of the cropped view the model trains on:

<p align="center">
 <a href="https://drive.google.com/open?id=1sgx-NqTc2hF2SouwBNriwsnOl3GIV3rG"><img src="./writeup/run1_trim.gif" alt="Overview"></a>
 <br>Click for full video.
</p>

### Track 2

Here's an example of the cropped view the model trains on:

<p align="center">
 <a href="https://drive.google.com/open?id=1wdsdsvJshJbwwyccqSz2inF7ktNjactb"><img src="./writeup/run2.gif" alt="Overview"></a>
 <br>Click for full video.
</p>

## Discussion

The current network has trouble driving straight without make slight adjustments  to the left or right. This problem could be attributed to collecting data with the arrow keys instead of a controller or joystick. I made a lot of adjustments while going through the tracks in an effort to stay in the center of the lane 😅

Another challenge is the high risk of overfitting with the limited data set. I've tried to counteract the overfitting by including L2 regularization and data augmentation by flipping the images. 