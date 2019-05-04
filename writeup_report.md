# **Behavioral Cloning** 

## Goals

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/losses.png "Train Validation Losses"
[image2]: ./images/center_driving.png "Center Driving"
[image3]: ./images/correction_left.png "Left Recovery Image"
[image4]: ./images/correction_right.png "Right Recovery Image"
[image5]: ./images/normal.png "Normal Image"
[image6]: ./images/flipped.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* `model.py` containing the script to create and train the model
* `plot_losses.py` containing the script for visualizing the training/validation losses
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network 
* `writeup_report.md` summarizing the results
* YouTube video clips: [Cockpit View](https://youtu.be/FDEj66vMGmo), [Road View](https://youtu.be/ofCpu5ntqvI) to demonstrate successful autonomous driving on the first track

#### 2. Submission includes functional code
Using the Udacity provided simulator and `drive.py` file, my model `model.h5` can autonomously drive the car around the track by executing
 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy Overview

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5 convolution layers and 4 dense layers (`model.py` lines 65-82, `build_model()`)

The model includes RELU layers to introduce nonlinearity, and the data is normalized and cropped in the model using Keras `Lambda` layer and `Cropping2D` layer (code line 68, 69). 

#### 2. Attempts to reduce overfitting in the model

The model contains `Dropout` layers in order to reduce overfitting (`model.py` lines 76, 78). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 88, 89). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an Adam optimizer so the learning rate was not tuned manually (model.py line 92). Callbacks including `EarlyStopping` and `ReduceLROnPlateau` were also used to facilitate generalization (code line 95-101).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I mainly used the [sample training data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) provided in the workspace. This dataset contains a combination of center lane driving, recovering from the left and right sides of the road. I also collected additional training data from my own driving. For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy Details

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to map the raw pixels from a front-facing camera to the steering commands for a self-driving car.

My first step was to use a convolution neural network model similar to the architecture published by NVDIA ([link](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)). I thought this model might be appropriate because it was proven to work well in the real world.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. To combat the overfitting, I modified the model to include two `Dropout` layers after two dense layers.

Then I also used orthogonal techniques such as early-stopping and learning-rate decay on plateau to further facilitate optimization and generalization. The losses from training and validation are plotted below. Though there is still some level of overfitting, the trained model performs well in test (autonomous driving in simulator).

![alt text][image1]

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road (shown in `video.mp4` and YouTube video clips: [Cockpit View](https://youtu.be/FDEj66vMGmo), [Road View](https://youtu.be/ofCpu5ntqvI)).

#### 2. Final Model Architecture

The final model architecture (`mode.py` lines 65-82, `build_model()`) consisted of a convolution neural network with the following layers and layer sizes

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 158, 24)       1824
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 77, 36)        21636
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 37, 48)         43248
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928
_________________________________________________________________
flatten_1 (Flatten)          (None, 2112)              0
_________________________________________________________________
dropout_1 (Dropout)          (None, 2112)              0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               211300
_________________________________________________________________
dropout_2 (Dropout)          (None, 100)               0
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11
=================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn correct from mistakes. These images show what a recovery looks like:

![alt text][image3]
![alt text][image4]

To augment the data sat, I also flipped images and angles thinking that this would facilitate generalization, as the track contains mostly left turns. For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]


After the collection process, I then preprocessed this data by normalizing the raw pixel values and cropping the image (code line 68, 69)

I finally randomly shuffled the data set and put 20% of the data into a validation set. I used Python generator to generate data for training rather than storing the training data in memory.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was determined by Early-Stopping. I used an Adam optimizer so that manually training the learning rate wasn't necessary. However, I also used `ReduceLROnPlateau` callback to decrease the learning rate when metric on the validation set plateaus.