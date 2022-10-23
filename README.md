# CIFAR-10 Classification using Convolutional Neural Networks (CNN)

<img width="500" src="images/banner.png">

## 1. Objective

The objective of this project is to develop a Convolutional Neural Network (CNN) to classify the 10 types of objects in the CIFAR-10 dataset.

## 2. Motivation

The CIFAR-10 dataset is another rich and labelled dataset used for benchmarking machine and deep learning image classification algorithms. It contains 60,000 images of typical objects such as animals, vehicles, airplanes, etc. 

In this section, we shall demonstrate how to develop convolutional neural network for clothing items classification from scratch, using the CIFAR-10dataset, including:
* How to prepare the input training and test data 
* How to deploy the model
* How to use the trained model to make predictions
* How to evaluate its performance

## 3. Data

A high-level description of the CIFAR-10 dataset is as follows:
* It contains 60000 color images, with size 32x32 pixels, 
* The objects in the images are in the following10 classes:
  * Airplane
  * Automobile
  * Bird
  * Cat
  * Deer
  * Dog
  * Frog
  * Horse
  * Ship
  * Truck.
* It has 6000 images per class:
  * There are 50000 training images
  * There are 10000 test images.
* Sample images from the CIFAR-10 data set are illustrated in the next figure:
  * There are significant variations between the different types of classes
  * There are significant variations as well as similarities between different examples of the same class.
  * Additional information about the CIFAR-10 dataset can be found in [1].
  
  <img width="500" src="images/CIFAR-10-sample-images.png">
  
## 4. Development

In this section, we shall demonstrate how to develop a Convolutional Neural Network (CNN) for clothing-articles classification from scratch, including:

# How to prepare the input training and test data 
# How to deploy the model
* How to use the trained model to make predictions
* How to evaluate its performance

* Author: Mohsen Ghazel (mghazel)
( Date: April 6th, 2021

* Project: CIFAR-10 Classification using Convolutional Neural Networks (CNN):


The objective of this project is to demonstrate how to develop a Convolutional Neural Network (CNN) to classify images from 10 different typical object classes, using the CIFAR-10 dataset:

* A high-level description of the CIFAR-10 dataset is as follows:

  * It contains 60000 color images, with size 32x32 pixels,
    * The objects in the images are in the following10 classes:
      * Airplane
      * Automobile
      * Bird
      * Cat
      * Deer
      * Dog
      * Frog
      * Horse
      * Ship
      * Truck.
  * It has 6000 images per class.
    * There are 50000 training images
    * There are 10000 test images
    * Additional detailed about the Fashion can be found here:
     * https://www.cs.toronto.edu/~kriz/cifar.html
     
* The CNN model training and evaluation process incvolves the following steps:
  1. Load the CIFAR-10 dataset of handwritten digits
  2. Build a simple CNN model
  3. Train the selected ML model
  4. Deploy the trained on the test data
  5. Evaluate the performance of the trained model using evaluation metrics:
     * Accuracy
     * Confusion Matrix
     * Other metrics derived form the confusion matrix.

### 4.1. Part 1: Imports and global variables:

#### 4.1.1. Standard scientific Python imports:


<pre style="color:#000000;bbackground:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Numpy</span>
<span style="color:#800000; font-weight:bold; ">import</span> numpy <span style="color:#800000; font-weight:bold; ">as</span> np
<span style="color:#696969; "># matplotlib</span>
<span style="color:#800000; font-weight:bold; ">import</span> matplotlib<span style="color:#808030; ">.</span>pyplot <span style="color:#800000; font-weight:bold; ">as</span> plt
<span style="color:#696969; "># - import sklearn to use the confusion matrix function</span>
<span style="color:#800000; font-weight:bold; ">from</span> sklearn<span style="color:#808030; ">.</span>metrics <span style="color:#800000; font-weight:bold; ">import</span> confusion_matrix
<span style="color:#696969; "># import itertools</span>
<span style="color:#800000; font-weight:bold; ">import</span> itertools
<span style="color:#696969; "># opencv</span>
<span style="color:#800000; font-weight:bold; ">import</span> cv2
<span style="color:#696969; "># tensorflow</span>
<span style="color:#800000; font-weight:bold; ">import</span> tensorflow <span style="color:#800000; font-weight:bold; ">as</span> tf

<span style="color:#696969; "># keras input layer</span>
<span style="color:#800000; font-weight:bold; ">from</span> tensorflow<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers <span style="color:#800000; font-weight:bold; ">import</span> <span style="color:#400000; ">Input</span>
<span style="color:#696969; "># keras conv2D layer</span>
<span style="color:#800000; font-weight:bold; ">from</span> tensorflow<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers <span style="color:#800000; font-weight:bold; ">import</span> Conv2D
<span style="color:#696969; "># keras MaxPooling2D layer</span>
<span style="color:#800000; font-weight:bold; ">from</span> tensorflow<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers <span style="color:#800000; font-weight:bold; ">import</span> MaxPooling2D
<span style="color:#696969; "># keras Dense layer</span>
<span style="color:#800000; font-weight:bold; ">from</span> tensorflow<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers <span style="color:#800000; font-weight:bold; ">import</span> Dense
<span style="color:#696969; "># keras Flatten layer</span>
<span style="color:#800000; font-weight:bold; ">from</span> tensorflow<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers <span style="color:#800000; font-weight:bold; ">import</span> Flatten
<span style="color:#696969; "># keras Dropout layer</span>
<span style="color:#800000; font-weight:bold; ">from</span> tensorflow<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers <span style="color:#800000; font-weight:bold; ">import</span> Dropout
<span style="color:#696969; "># batch-normalization</span>
<span style="color:#800000; font-weight:bold; ">from</span> tensorflow<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers <span style="color:#800000; font-weight:bold; ">import</span> BatchNormalization
<span style="color:#696969; "># global-max-pooling</span>
<span style="color:#800000; font-weight:bold; ">from</span> tensorflow<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers <span style="color:#800000; font-weight:bold; ">import</span> GlobalMaxPooling2D

<span style="color:#696969; "># keras model</span>
<span style="color:#800000; font-weight:bold; ">from</span> tensorflow<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>models <span style="color:#800000; font-weight:bold; ">import</span> Model
<span style="color:#696969; "># keras sequential model</span>
<span style="color:#800000; font-weight:bold; ">from</span> tensorflow<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>models <span style="color:#800000; font-weight:bold; ">import</span> Sequential
<span style="color:#696969; "># optimizers</span>
<span style="color:#800000; font-weight:bold; ">from</span> tensorflow<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>optimizers <span style="color:#800000; font-weight:bold; ">import</span> SGD

<span style="color:#696969; "># random number generators values</span>
<span style="color:#696969; "># seed for reproducing the random number generation</span>
<span style="color:#800000; font-weight:bold; ">from</span> random <span style="color:#800000; font-weight:bold; ">import</span> seed
<span style="color:#696969; "># random integers: I(0,M)</span>
<span style="color:#800000; font-weight:bold; ">from</span> random <span style="color:#800000; font-weight:bold; ">import</span> randint
<span style="color:#696969; "># random standard unform: U(0,1)</span>
<span style="color:#800000; font-weight:bold; ">from</span> random <span style="color:#800000; font-weight:bold; ">import</span> random
<span style="color:#696969; "># time</span>
<span style="color:#800000; font-weight:bold; ">import</span> datetime
<span style="color:#696969; "># I/O</span>
<span style="color:#800000; font-weight:bold; ">import</span> os
<span style="color:#696969; "># sys</span>
<span style="color:#800000; font-weight:bold; ">import</span> sys

<span style="color:#696969; "># check for successful package imports and versions</span>
<span style="color:#696969; "># python</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Python version : {0} "</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>sys<span style="color:#808030; ">.</span>version<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># OpenCV</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"OpenCV version : {0} "</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>cv2<span style="color:#808030; ">.</span>__version__<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># numpy</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Numpy version  : {0}"</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>np<span style="color:#808030; ">.</span>__version__<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># tensorflow</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Tensorflow version  : {0}"</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>tf<span style="color:#808030; ">.</span>__version__<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>

Python version <span style="color:#808030; ">:</span> <span style="color:#008000; ">3.7</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">10</span> <span style="color:#808030; ">(</span>default<span style="color:#808030; ">,</span> Feb <span style="color:#008c00; ">20</span> <span style="color:#008c00; ">2021</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">21</span><span style="color:#808030; ">:</span><span style="color:#008c00; ">17</span><span style="color:#808030; ">:</span><span style="color:#008c00; ">23</span><span style="color:#808030; ">)</span> 
<span style="color:#808030; ">[</span>GCC <span style="color:#008000; ">7.5</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span> 
OpenCV version <span style="color:#808030; ">:</span> <span style="color:#008000; ">4.1</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">2</span> 
Numpy version  <span style="color:#808030; ">:</span> <span style="color:#008000; ">1.19</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">5</span>
Tensorflow version  <span style="color:#808030; ">:</span> <span style="color:#008000; ">2.4</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">1</span>
</pre>


#### 4.1.2. Global variables:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># -set the random_state seed = 100 for reproducibilty</span>
random_state_seed <span style="color:#808030; ">=</span> <span style="color:#008c00; ">100</span>

<span style="color:#696969; "># the number of visualized images</span>
num_visualized_images <span style="color:#808030; ">=</span> <span style="color:#008c00; ">25</span>
</pre>


### 4.2. Part 2: Load CIFAR-10 Dataset

#### 4.2.1. Load the CIFAR-10dataset:

* Load the CIFAR-10T dataset of clothing-articles:
  * A high-level description of the CIFAR-10 dataset is as follows:
    * It contains 60000 color images, with size 32x32 pixels,
      * The objects in the images are in the following10 classes:
        * Airplane
        * Automobile
        * Bird
        * Cat
        * Deer
        * Dog
        * Frog
        * Horse
        * Ship
        * Truck.
    * It has 6000 images per class.
      * There are 50000 training images
      * There are 10000 test images
      * Additional detailed about the Fashion can be found here:
       * https://www.cs.toronto.edu/~kriz/cifar.html


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Load in the CIFAR10 data set</span>
<span style="color:#696969; "># - It has 10 classes</span>
cifar10 <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>datasets<span style="color:#808030; ">.</span>cifar10
<span style="color:#696969; "># extract the training and testing subsets</span>
<span style="color:#808030; ">(</span>x_train<span style="color:#808030; ">,</span> y_train<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span>x_test<span style="color:#808030; ">,</span> y_test<span style="color:#808030; ">)</span> <span style="color:#808030; ">=</span> cifar10<span style="color:#808030; ">.</span>load_data<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>

Downloading data <span style="color:#800000; font-weight:bold; ">from</span> https<span style="color:#808030; ">:</span><span style="color:#44aadd; ">//</span>www<span style="color:#808030; ">.</span>cs<span style="color:#808030; ">.</span>toronto<span style="color:#808030; ">.</span>edu<span style="color:#44aadd; ">/</span><span style="color:#44aadd; ">~</span>kriz<span style="color:#44aadd; ">/</span>cifar<span style="color:#44aadd; ">-</span><span style="color:#008c00; ">10</span><span style="color:#44aadd; ">-</span>python<span style="color:#808030; ">.</span>tar<span style="color:#808030; ">.</span>gz
<span style="color:#008c00; ">170500096</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">170498071</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">2</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">0</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">us</span><span style="color:#44aadd; ">/</span>step
</pre>

#### 4.2.2. Explore training and test images:

##### 4.2.2.1 Display the number and shape of the training and test subsets:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># Training data:</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># the number of training images</span>
num_train_images <span style="color:#808030; ">=</span> x_train<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Training data:"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"x_train.shape: "</span><span style="color:#808030; ">,</span> x_train<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Number of training images: "</span><span style="color:#808030; ">,</span> num_train_images<span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Image size: "</span><span style="color:#808030; ">,</span> x_train<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">:</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>

<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># Test data:</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># the number of test images</span>
num_test_images <span style="color:#808030; ">=</span> x_test<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Test data:"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"x_test.shape: "</span><span style="color:#808030; ">,</span> x_test<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Number of test images: "</span><span style="color:#808030; ">,</span> num_test_images<span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Image size: "</span><span style="color:#808030; ">,</span> x_test<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">:</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>

<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Training data<span style="color:#808030; ">:</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
x_train<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">:</span>  <span style="color:#808030; ">(</span><span style="color:#008c00; ">50000</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span>
Number of training images<span style="color:#808030; ">:</span>  <span style="color:#008c00; ">50000</span>
Image size<span style="color:#808030; ">:</span>  <span style="color:#808030; ">(</span><span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Test data<span style="color:#808030; ">:</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
x_test<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">:</span>  <span style="color:#808030; ">(</span><span style="color:#008c00; ">10000</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span>
Number of test images<span style="color:#808030; ">:</span>  <span style="color:#008c00; ">10000</span>
Image size<span style="color:#808030; ">:</span>  <span style="color:#808030; ">(</span><span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
</pre>


##### 4.2.2.2. Reshape the training and test target vectors:

* Convert the y_train and y_test to 1-dimensional vectors:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># flatten the y_train targets vector to convert to 1-dimensional</span>
y_train <span style="color:#808030; ">=</span> y_train<span style="color:#808030; ">.</span>flatten<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>

<span style="color:#696969; "># flatten the y_test targets to convert to 1-dimensional</span>
y_test <span style="color:#808030; ">=</span> y_test<span style="color:#808030; ">.</span>flatten<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
</pre>

##### 4.2.2.3. Display the targets/classes:

* There 10 classes:
  * Each training and test example is assigned to one of the following labels:

<img width="100" src="images/CIFAR10 -10-classes-labels.PNG">

###### 4.2.2.3.1. Display the number of classes:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Display the number of classes:</span>
num_classes <span style="color:#808030; ">=</span> <span style="color:#400000; ">len</span><span style="color:#808030; ">(</span><span style="color:#400000; ">set</span><span style="color:#808030; ">(</span>y_train<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'-------------------------------------------------'</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"The number of classes in the CIFAR dataset = "</span><span style="color:#808030; ">,</span> num_classes<span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'-------------------------------------------------'</span><span style="color:#808030; ">)</span>

<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
The number of classes <span style="color:#800000; font-weight:bold; ">in</span> the CIFAR dataset <span style="color:#808030; ">=</span>  <span style="color:#008c00; ">10</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
</pre>

###### 4.2.2.3.2. Create meaningful labels for the different classes:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># the labels mapping</span>
labels <span style="color:#808030; ">=</span> <span style="color:#696969; ">'''airplane</span>
<span style="color:#696969; ">automobile</span>
<span style="color:#696969; ">bird</span>
<span style="color:#696969; ">cat</span>
<span style="color:#696969; ">deer</span>
<span style="color:#696969; ">dog</span>
<span style="color:#696969; ">frog</span>
<span style="color:#696969; ">horse</span>
<span style="color:#696969; ">ship</span>
<span style="color:#696969; ">truck'''</span><span style="color:#808030; ">.</span>split<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># display the class labels</span>
<span style="color:#800000; font-weight:bold; ">for</span> counter <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span>num_classes<span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
  <span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Class ID = {}, Class name = {}'</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>counter<span style="color:#808030; ">,</span> labels<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>

<span style="color:#800000; font-weight:bold; ">Class</span> <span style="color:#400000; ">ID</span> <span style="color:#808030; ">=</span> <span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span> <span style="color:#800000; font-weight:bold; ">Class</span> name <span style="color:#808030; ">=</span> airplane
<span style="color:#800000; font-weight:bold; ">Class</span> <span style="color:#400000; ">ID</span> <span style="color:#808030; ">=</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">,</span> <span style="color:#800000; font-weight:bold; ">Class</span> name <span style="color:#808030; ">=</span> automobile
<span style="color:#800000; font-weight:bold; ">Class</span> <span style="color:#400000; ">ID</span> <span style="color:#808030; ">=</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span> <span style="color:#800000; font-weight:bold; ">Class</span> name <span style="color:#808030; ">=</span> bird
<span style="color:#800000; font-weight:bold; ">Class</span> <span style="color:#400000; ">ID</span> <span style="color:#808030; ">=</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span> <span style="color:#800000; font-weight:bold; ">Class</span> name <span style="color:#808030; ">=</span> cat
<span style="color:#800000; font-weight:bold; ">Class</span> <span style="color:#400000; ">ID</span> <span style="color:#808030; ">=</span> <span style="color:#008c00; ">4</span><span style="color:#808030; ">,</span> <span style="color:#800000; font-weight:bold; ">Class</span> name <span style="color:#808030; ">=</span> deer
<span style="color:#800000; font-weight:bold; ">Class</span> <span style="color:#400000; ">ID</span> <span style="color:#808030; ">=</span> <span style="color:#008c00; ">5</span><span style="color:#808030; ">,</span> <span style="color:#800000; font-weight:bold; ">Class</span> name <span style="color:#808030; ">=</span> dog
<span style="color:#800000; font-weight:bold; ">Class</span> <span style="color:#400000; ">ID</span> <span style="color:#808030; ">=</span> <span style="color:#008c00; ">6</span><span style="color:#808030; ">,</span> <span style="color:#800000; font-weight:bold; ">Class</span> name <span style="color:#808030; ">=</span> frog
<span style="color:#800000; font-weight:bold; ">Class</span> <span style="color:#400000; ">ID</span> <span style="color:#808030; ">=</span> <span style="color:#008c00; ">7</span><span style="color:#808030; ">,</span> <span style="color:#800000; font-weight:bold; ">Class</span> name <span style="color:#808030; ">=</span> horse
<span style="color:#800000; font-weight:bold; ">Class</span> <span style="color:#400000; ">ID</span> <span style="color:#808030; ">=</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">,</span> <span style="color:#800000; font-weight:bold; ">Class</span> name <span style="color:#808030; ">=</span> ship
<span style="color:#800000; font-weight:bold; ">Class</span> <span style="color:#400000; ">ID</span> <span style="color:#808030; ">=</span> <span style="color:#008c00; ">9</span><span style="color:#808030; ">,</span> <span style="color:#800000; font-weight:bold; ">Class</span> name <span style="color:#808030; ">=</span> truck
</pre>


##### 4.2.2.4. Examine the number of images for each class of the training and testing subsets:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Create a histogram of the number of images in each class/digit:</span>
<span style="color:#800000; font-weight:bold; ">def</span> plot_bar<span style="color:#808030; ">(</span>y<span style="color:#808030; ">,</span> loc<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'left'</span><span style="color:#808030; ">,</span> relative<span style="color:#808030; ">=</span><span style="color:#074726; ">True</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
    width <span style="color:#808030; ">=</span> <span style="color:#008000; ">0.35</span>
    <span style="color:#800000; font-weight:bold; ">if</span> loc <span style="color:#44aadd; ">==</span> <span style="color:#0000e6; ">'left'</span><span style="color:#808030; ">:</span>
        n <span style="color:#808030; ">=</span> <span style="color:#44aadd; ">-</span><span style="color:#008000; ">0.5</span>
    <span style="color:#800000; font-weight:bold; ">elif</span> loc <span style="color:#44aadd; ">==</span> <span style="color:#0000e6; ">'right'</span><span style="color:#808030; ">:</span>
        n <span style="color:#808030; ">=</span> <span style="color:#008000; ">0.5</span>
     
    <span style="color:#696969; "># calculate counts per type and sort, to ensure their order</span>
    unique<span style="color:#808030; ">,</span> counts <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>unique<span style="color:#808030; ">(</span>y<span style="color:#808030; ">,</span> return_counts<span style="color:#808030; ">=</span><span style="color:#074726; ">True</span><span style="color:#808030; ">)</span>
    sorted_index <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>argsort<span style="color:#808030; ">(</span>unique<span style="color:#808030; ">)</span>
    unique <span style="color:#808030; ">=</span> unique<span style="color:#808030; ">[</span>sorted_index<span style="color:#808030; ">]</span>
     
    <span style="color:#800000; font-weight:bold; ">if</span> relative<span style="color:#808030; ">:</span>
        <span style="color:#696969; "># plot as a percentage</span>
        counts <span style="color:#808030; ">=</span> <span style="color:#008c00; ">100</span><span style="color:#44aadd; ">*</span>counts<span style="color:#808030; ">[</span>sorted_index<span style="color:#808030; ">]</span><span style="color:#44aadd; ">/</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>y<span style="color:#808030; ">)</span>
        ylabel_text <span style="color:#808030; ">=</span> <span style="color:#0000e6; ">'% count'</span>
    <span style="color:#800000; font-weight:bold; ">else</span><span style="color:#808030; ">:</span>
        <span style="color:#696969; "># plot counts</span>
        counts <span style="color:#808030; ">=</span> counts<span style="color:#808030; ">[</span>sorted_index<span style="color:#808030; ">]</span>
        ylabel_text <span style="color:#808030; ">=</span> <span style="color:#0000e6; ">'count'</span>
         
    xtemp <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>arange<span style="color:#808030; ">(</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>unique<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
    plt<span style="color:#808030; ">.</span>bar<span style="color:#808030; ">(</span>xtemp <span style="color:#44aadd; ">+</span> n<span style="color:#44aadd; ">*</span>width<span style="color:#808030; ">,</span> counts<span style="color:#808030; ">,</span> align<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'center'</span><span style="color:#808030; ">,</span> alpha<span style="color:#808030; ">=</span><span style="color:#008000; ">.7</span><span style="color:#808030; ">,</span> width<span style="color:#808030; ">=</span>width<span style="color:#808030; ">)</span>
    plt<span style="color:#808030; ">.</span>xticks<span style="color:#808030; ">(</span>xtemp<span style="color:#808030; ">,</span> unique<span style="color:#808030; ">,</span> rotation<span style="color:#808030; ">=</span><span style="color:#008c00; ">45</span><span style="color:#808030; ">)</span>
    plt<span style="color:#808030; ">.</span>xlabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'digit'</span><span style="color:#808030; ">)</span>
    plt<span style="color:#808030; ">.</span>ylabel<span style="color:#808030; ">(</span>ylabel_text<span style="color:#808030; ">)</span>
 
plt<span style="color:#808030; ">.</span>suptitle<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Frequency of images per digit'</span><span style="color:#808030; ">)</span>
plot_bar<span style="color:#808030; ">(</span>y_train<span style="color:#808030; ">,</span> loc<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'left'</span><span style="color:#808030; ">)</span>
plot_bar<span style="color:#808030; ">(</span>y_test<span style="color:#808030; ">,</span> loc<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'right'</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>legend<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span>
    <span style="color:#0000e6; ">'train ({0} images)'</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>y_train<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> 
    <span style="color:#0000e6; ">'test ({0} images)'</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>y_test<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> 
<span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
</pre>

<img width="500" src="images/CIFAR10-10-classes-examples-distribution.png">

##### 4.2.2.5. Visualize some of the training and test images and their associated targets:

* First implement a visualization functionality to visualize the number of randomly selected images:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">"""</span>
<span style="color:#696969; "># A utility function to visualize multiple images:</span>
<span style="color:#696969; ">"""</span>
<span style="color:#800000; font-weight:bold; ">def</span> visualize_images_and_labels<span style="color:#808030; ">(</span>num_visualized_images <span style="color:#808030; ">=</span> <span style="color:#008c00; ">25</span><span style="color:#808030; ">,</span> dataset_flag <span style="color:#808030; ">=</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
  <span style="color:#696969; ">"""To visualize images.</span>
<span style="color:#696969; "></span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Keyword arguments:</span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- num_visualized_images -- the number of visualized images (deafult 25)</span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- dataset_flag -- 1: training dataset, 2: test dataset</span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Return:</span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- None</span>
<span style="color:#696969; ">&nbsp;&nbsp;"""</span>
  <span style="color:#696969; ">#--------------------------------------------</span>
  <span style="color:#696969; "># the suplot grid shape:</span>
  <span style="color:#696969; ">#--------------------------------------------</span>
  num_rows <span style="color:#808030; ">=</span> <span style="color:#008c00; ">5</span>
  <span style="color:#696969; "># the number of columns</span>
  num_cols <span style="color:#808030; ">=</span> num_visualized_images <span style="color:#44aadd; ">//</span> num_rows
  <span style="color:#696969; "># setup the subplots axes</span>
  fig<span style="color:#808030; ">,</span> axes <span style="color:#808030; ">=</span> plt<span style="color:#808030; ">.</span>subplots<span style="color:#808030; ">(</span>nrows<span style="color:#808030; ">=</span>num_rows<span style="color:#808030; ">,</span> ncols<span style="color:#808030; ">=</span>num_cols<span style="color:#808030; ">,</span> figsize<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">8</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">10</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
  <span style="color:#696969; "># set a seed random number generator for reproducible results</span>
  seed<span style="color:#808030; ">(</span>random_state_seed<span style="color:#808030; ">)</span>
  <span style="color:#696969; "># iterate over the sub-plots</span>
  <span style="color:#800000; font-weight:bold; ">for</span> row <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span>num_rows<span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
      <span style="color:#800000; font-weight:bold; ">for</span> col <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span>num_cols<span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
        <span style="color:#696969; "># get the next figure axis</span>
        ax <span style="color:#808030; ">=</span> axes<span style="color:#808030; ">[</span>row<span style="color:#808030; ">,</span> col<span style="color:#808030; ">]</span><span style="color:#808030; ">;</span>
        <span style="color:#696969; "># turn-off subplot axis</span>
        ax<span style="color:#808030; ">.</span>set_axis_off<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        <span style="color:#696969; "># if the dataset_flag = 1: Training data set</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        <span style="color:#800000; font-weight:bold; ">if</span> <span style="color:#808030; ">(</span> dataset_flag <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">1</span> <span style="color:#808030; ">)</span><span style="color:#808030; ">:</span> 
          <span style="color:#696969; "># generate a random image counter</span>
          counter <span style="color:#808030; ">=</span> randint<span style="color:#808030; ">(</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span>num_train_images<span style="color:#808030; ">)</span>
          <span style="color:#696969; "># get the training image</span>
          image <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>squeeze<span style="color:#808030; ">(</span>x_train<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">,</span><span style="color:#808030; ">:</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
          <span style="color:#696969; "># get the target associated with the image</span>
          label <span style="color:#808030; ">=</span> labels<span style="color:#808030; ">[</span>y_train<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">]</span><span style="color:#808030; ">]</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        <span style="color:#696969; "># dataset_flag = 2: Test data set</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        <span style="color:#800000; font-weight:bold; ">else</span><span style="color:#808030; ">:</span> 
          <span style="color:#696969; "># generate a random image counter</span>
          counter <span style="color:#808030; ">=</span> randint<span style="color:#808030; ">(</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span>num_test_images<span style="color:#808030; ">)</span>
          <span style="color:#696969; "># get the test image</span>
          image <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>squeeze<span style="color:#808030; ">(</span>x_test<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">,</span><span style="color:#808030; ">:</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
          <span style="color:#696969; "># get the target associated with the image</span>
          label <span style="color:#808030; ">=</span> labels<span style="color:#808030; ">[</span>y_test<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">]</span><span style="color:#808030; ">]</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        <span style="color:#696969; "># display the image</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        ax<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>image<span style="color:#808030; ">,</span> cmap<span style="color:#808030; ">=</span>plt<span style="color:#808030; ">.</span>cm<span style="color:#808030; ">.</span>gray_r<span style="color:#808030; ">,</span> interpolation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'nearest'</span><span style="color:#808030; ">)</span>
        <span style="color:#696969; "># set the title showing the image label</span>
        ax<span style="color:#808030; ">.</span>set_title<span style="color:#808030; ">(</span><span style="color:#400000; ">str</span><span style="color:#808030; ">(</span>label<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> size <span style="color:#808030; ">=</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">)</span>
</pre>


##### 4.2.2.5.1. Visualize some of the training images and their associated targets:

<img width="500" src="images/CIFAR10-25-train-images.png">

##### 4.2.2.5.2. Visualize some of the test images and their associated targets:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># call the function to visualize the training images</span>
visualize_images_and_labels<span style="color:#808030; ">(</span>num_visualized_images<span style="color:#808030; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span>
</pre>

<img width="500" src="images/CIFAR10-25-test-images.png">

#### 4.2.3. Normalize the training and test images to the interval: [0, 1]:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Normalize the training images</span>
x_train <span style="color:#808030; ">=</span> x_train <span style="color:#44aadd; ">/</span> <span style="color:#008000; ">255.0</span>
<span style="color:#696969; "># Normalize the test images</span>
x_test <span style="color:#808030; ">=</span> x_test <span style="color:#44aadd; ">/</span> <span style="color:#008000; ">255.0</span>
</pre>


### 4.3. Part 3: Build the CNN model architecture

#### 4.3.1. Design the structure of the CNN model to classify the CIFAR-10 images:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Build the sequential CNN model</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Build the model using the functional API</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 1: Input layer</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - input images size: (28, 28, 3)</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
i <span style="color:#808030; ">=</span> <span style="color:#400000; ">Input</span><span style="color:#808030; ">(</span>shape<span style="color:#808030; ">=</span>x_train<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">.</span>shape<span style="color:#808030; ">)</span>   
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 2: Convolutional layer</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - 32 filters:  </span>
<span style="color:#696969; ">#   - size: 3x3</span>
<span style="color:#696969; ">#   - same</span>
<span style="color:#696969; ">#   - stride = 1 </span>
<span style="color:#696969; "># - Activation function: relu</span>
<span style="color:#696969; ">#------------------------------------------------------------------------------- </span>
x <span style="color:#808030; ">=</span> Conv2D<span style="color:#808030; ">(</span><span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">,</span> padding<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'same'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>i<span style="color:#808030; ">)</span> 
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 3: Batch normalization</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------              </span>
x <span style="color:#808030; ">=</span> BatchNormalization<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>x<span style="color:#808030; ">)</span>   
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 4: Convolutional layer</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - 32 filters:  </span>
<span style="color:#696969; ">#   - size: 3x3</span>
<span style="color:#696969; ">#   - same</span>
<span style="color:#696969; ">#   - stride = 1 </span>
<span style="color:#696969; "># - Activation function: relu</span>
<span style="color:#696969; ">#------------------------------------------------------------------------------- </span>
x <span style="color:#808030; ">=</span> Conv2D<span style="color:#808030; ">(</span><span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">,</span> padding<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'same'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>x<span style="color:#808030; ">)</span> 
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 5: Batch normalization</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------              </span>
x <span style="color:#808030; ">=</span> BatchNormalization<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>x<span style="color:#808030; ">)</span>  
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 6: Max-pooling layer</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - Max-pooling  </span>
<span style="color:#696969; ">#   - size: 2x2</span>
<span style="color:#696969; ">#------------------------------------------------------------------------------- </span>
x <span style="color:#808030; ">=</span> MaxPooling2D<span style="color:#808030; ">(</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>x<span style="color:#808030; ">)</span>   
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 7: Convolutional layer</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - 32 filters:  </span>
<span style="color:#696969; ">#   - size: 3x3</span>
<span style="color:#696969; ">#   - same</span>
<span style="color:#696969; ">#   - stride = 1 </span>
<span style="color:#696969; "># - Activation function: relu</span>
<span style="color:#696969; ">#------------------------------------------------------------------------------- </span>
x <span style="color:#808030; ">=</span> Conv2D<span style="color:#808030; ">(</span><span style="color:#008c00; ">64</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">,</span> padding<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'same'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>x<span style="color:#808030; ">)</span> 
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 8: Batch normalization</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------   </span>
x <span style="color:#808030; ">=</span> BatchNormalization<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>x<span style="color:#808030; ">)</span>                                     
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 9: Convolutional layer</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - 64 filters:  </span>
<span style="color:#696969; ">#   - size: 3x3</span>
<span style="color:#696969; ">#   - same</span>
<span style="color:#696969; ">#   - stride = 1 </span>
<span style="color:#696969; "># - Activation function: relu</span>
<span style="color:#696969; ">#------------------------------------------------------------------------------- </span>
x <span style="color:#808030; ">=</span> Conv2D<span style="color:#808030; ">(</span><span style="color:#008c00; ">64</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">,</span> padding<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'same'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>x<span style="color:#808030; ">)</span>   
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 10: Batch normalization</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------    </span>
x <span style="color:#808030; ">=</span> BatchNormalization<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>x<span style="color:#808030; ">)</span>        
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 11: Max-pooling layer</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - Max-pooling  </span>
<span style="color:#696969; ">#   - size: 2x2</span>
<span style="color:#696969; ">#------------------------------------------------------------------------------- </span>
x <span style="color:#808030; ">=</span> MaxPooling2D<span style="color:#808030; ">(</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>x<span style="color:#808030; ">)</span>                               
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 12: Convolutional layer</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - 128 filters:  </span>
<span style="color:#696969; ">#   - size: 3x3</span>
<span style="color:#696969; ">#   - same</span>
<span style="color:#696969; ">#   - stride = 1 </span>
<span style="color:#696969; "># - Activation function: relu</span>
<span style="color:#696969; ">#------------------------------------------------------------------------------- </span>
x <span style="color:#808030; ">=</span> Conv2D<span style="color:#808030; ">(</span><span style="color:#008c00; ">128</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">,</span> padding<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'same'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>x<span style="color:#808030; ">)</span>  
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 13: Batch normalization</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------    </span>
x <span style="color:#808030; ">=</span> BatchNormalization<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>x<span style="color:#808030; ">)</span>  
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 14: Convolutional layer</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - 128 filters:  </span>
<span style="color:#696969; ">#   - size: 3x3</span>
<span style="color:#696969; ">#   - same</span>
<span style="color:#696969; ">#   - stride = 1 </span>
<span style="color:#696969; "># - Activation function: relu</span>
<span style="color:#696969; ">#------------------------------------------------------------------------------- </span>
x <span style="color:#808030; ">=</span> Conv2D<span style="color:#808030; ">(</span><span style="color:#008c00; ">128</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">,</span> padding<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'same'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>x<span style="color:#808030; ">)</span>   
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 15: Batch normalization</span>
<span style="color:#696969; ">#------------------------------------------------------------------------------- </span>
x <span style="color:#808030; ">=</span> BatchNormalization<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>x<span style="color:#808030; ">)</span>    
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 16: Max-pooling layer</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - Max-pooling  </span>
<span style="color:#696969; ">#   - size: 2x2</span>
<span style="color:#696969; ">#------------------------------------------------------------------------------- </span>
x <span style="color:#808030; ">=</span> MaxPooling2D<span style="color:#808030; ">(</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>x<span style="color:#808030; ">)</span>        
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 17: Flatten</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - Flatten to connect to the next Fully-Connected Dense layer</span>
<span style="color:#696969; ">#------------------------------------------------------------------------------- </span>
x <span style="color:#808030; ">=</span> Flatten<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>x<span style="color:#808030; ">)</span>       
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 18: Dropout layer</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - p = 0.20  </span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------                                         </span>
x <span style="color:#808030; ">=</span> Dropout<span style="color:#808030; ">(</span><span style="color:#008000; ">0.2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>x<span style="color:#808030; ">)</span>      
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 19: Dense layer</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - 1024 neurons</span>
<span style="color:#696969; "># - Activation function: relu</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------                                    </span>
x <span style="color:#808030; ">=</span> Dense<span style="color:#808030; ">(</span><span style="color:#008c00; ">1024</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>x<span style="color:#808030; ">)</span>    
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 20: Dropout layer</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - p = 0.20  </span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------                        </span>
x <span style="color:#808030; ">=</span> Dropout<span style="color:#808030; ">(</span><span style="color:#008000; ">0.2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>x<span style="color:#808030; ">)</span>      
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 21: Output layer</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - Number of neurons: num_classes </span>
<span style="color:#696969; "># - Activation function: softmax:</span>
<span style="color:#696969; "># - Suitable for multi-class classification.</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------                                       </span>
x <span style="color:#808030; ">=</span> Dense<span style="color:#808030; ">(</span>num_classes<span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'softmax'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>x<span style="color:#808030; ">)</span>                         
<span style="color:#696969; ">#-------------------------------------------------------------------------------          </span>
<span style="color:#696969; "># Create the model with above structure:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
model <span style="color:#808030; ">=</span> Model<span style="color:#808030; ">(</span>i<span style="color:#808030; ">,</span> x<span style="color:#808030; ">)</span>
</pre>

#### 4.3.2. Print the designed model summary:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;">Model<span style="color:#808030; ">:</span> <span style="color:#0000e6; ">"model"</span>
_________________________________________________________________
Layer <span style="color:#808030; ">(</span><span style="color:#400000; ">type</span><span style="color:#808030; ">)</span>                 Output Shape              Param <span style="color:#696969; ">#   </span>
<span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span>
input_1 <span style="color:#808030; ">(</span>InputLayer<span style="color:#808030; ">)</span>         <span style="color:#808030; ">[</span><span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">]</span>       <span style="color:#008c00; ">0</span>         
_________________________________________________________________
conv2d <span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">)</span>              <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">)</span>        <span style="color:#008c00; ">896</span>       
_________________________________________________________________
batch_normalization <span style="color:#808030; ">(</span>BatchNo <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">)</span>        <span style="color:#008c00; ">128</span>       
_________________________________________________________________
conv2d_1 <span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">)</span>            <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">)</span>        <span style="color:#008c00; ">9248</span>      
_________________________________________________________________
batch_normalization_1 <span style="color:#808030; ">(</span>Batch <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">)</span>        <span style="color:#008c00; ">128</span>       
_________________________________________________________________
max_pooling2d <span style="color:#808030; ">(</span>MaxPooling2D<span style="color:#808030; ">)</span> <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">)</span>        <span style="color:#008c00; ">0</span>         
_________________________________________________________________
conv2d_2 <span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">)</span>            <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">)</span>        <span style="color:#008c00; ">18496</span>     
_________________________________________________________________
batch_normalization_2 <span style="color:#808030; ">(</span>Batch <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">)</span>        <span style="color:#008c00; ">256</span>       
_________________________________________________________________
conv2d_3 <span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">)</span>            <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">)</span>        <span style="color:#008c00; ">36928</span>     
_________________________________________________________________
batch_normalization_3 <span style="color:#808030; ">(</span>Batch <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">)</span>        <span style="color:#008c00; ">256</span>       
_________________________________________________________________
max_pooling2d_1 <span style="color:#808030; ">(</span>MaxPooling2 <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">)</span>          <span style="color:#008c00; ">0</span>         
_________________________________________________________________
conv2d_4 <span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">)</span>            <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">)</span>         <span style="color:#008c00; ">73856</span>     
_________________________________________________________________
batch_normalization_4 <span style="color:#808030; ">(</span>Batch <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">)</span>         <span style="color:#008c00; ">512</span>       
_________________________________________________________________
conv2d_5 <span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">)</span>            <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">)</span>         <span style="color:#008c00; ">147584</span>    
_________________________________________________________________
batch_normalization_5 <span style="color:#808030; ">(</span>Batch <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">)</span>         <span style="color:#008c00; ">512</span>       
_________________________________________________________________
max_pooling2d_2 <span style="color:#808030; ">(</span>MaxPooling2 <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">4</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">4</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">)</span>         <span style="color:#008c00; ">0</span>         
_________________________________________________________________
flatten <span style="color:#808030; ">(</span>Flatten<span style="color:#808030; ">)</span>            <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">2048</span><span style="color:#808030; ">)</span>              <span style="color:#008c00; ">0</span>         
_________________________________________________________________
dropout <span style="color:#808030; ">(</span>Dropout<span style="color:#808030; ">)</span>            <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">2048</span><span style="color:#808030; ">)</span>              <span style="color:#008c00; ">0</span>         
_________________________________________________________________
dense <span style="color:#808030; ">(</span>Dense<span style="color:#808030; ">)</span>                <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">1024</span><span style="color:#808030; ">)</span>              <span style="color:#008c00; ">2098176</span>   
_________________________________________________________________
dropout_1 <span style="color:#808030; ">(</span>Dropout<span style="color:#808030; ">)</span>          <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">1024</span><span style="color:#808030; ">)</span>              <span style="color:#008c00; ">0</span>         
_________________________________________________________________
dense_1 <span style="color:#808030; ">(</span>Dense<span style="color:#808030; ">)</span>              <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">10</span><span style="color:#808030; ">)</span>                <span style="color:#008c00; ">10250</span>     
<span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span>
Total params<span style="color:#808030; ">:</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">397</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">226</span>
Trainable params<span style="color:#808030; ">:</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">396</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">330</span>
Non<span style="color:#44aadd; ">-</span>trainable params<span style="color:#808030; ">:</span> <span style="color:#008c00; ">896</span>
_________________________________________________________________
</pre>


### 4.4. Part 4: Compile the CNN model

* Compile the CNN model, developed above:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Compile the model</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
model<span style="color:#808030; ">.</span><span style="color:#400000; ">compile</span><span style="color:#808030; ">(</span>optimizer<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'adam'</span><span style="color:#808030; ">,</span>                       <span style="color:#696969; "># optimzer: adam</span>
              loss<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'sparse_categorical_crossentropy'</span><span style="color:#808030; ">,</span> <span style="color:#696969; "># used for multi-class models</span>
              metrics<span style="color:#808030; ">=</span><span style="color:#808030; ">[</span><span style="color:#0000e6; ">'accuracy'</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>                   <span style="color:#696969; "># performance evaluation metric</span>
</pre>

### 4.5. Part 5: Train/Fit the model:

* Start training the compiled CNN model.


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Fit the model:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - set the number of epochs</span>
num_epochs <span style="color:#808030; ">=</span> <span style="color:#008c00; ">100</span>
<span style="color:#696969; "># train the model</span>
r <span style="color:#808030; ">=</span> model<span style="color:#808030; ">.</span>fit<span style="color:#808030; ">(</span>x_train<span style="color:#808030; ">,</span> y_train<span style="color:#808030; ">,</span> validation_data<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span>x_test<span style="color:#808030; ">,</span> y_test<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> epochs<span style="color:#808030; ">=</span>num_epochs<span style="color:#808030; ">)</span> metric

Epoch <span style="color:#008c00; ">1</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1563</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1563</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">44</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">7</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">1.7406</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.4430</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">1.1168</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.6064</span>
Epoch <span style="color:#008c00; ">2</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1563</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1563</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">10</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">6</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8931</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.6859</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">1.0089</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.6618</span>
Epoch <span style="color:#008c00; ">3</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1563</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1563</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">10</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">6</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.7057</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.7548</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8089</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.7254</span>
Epoch <span style="color:#008c00; ">4</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1563</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1563</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">10</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">6</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.5802</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.7983</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.6552</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.7797</span>
Epoch <span style="color:#008c00; ">5</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1563</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1563</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">10</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">6</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.4876</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8325</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.6397</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.7898</span>
<span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span>
<span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span>
Epoch <span style="color:#008c00; ">95</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1563</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1563</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">10</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">7</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0244</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9923</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">1.1665</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8421</span>
Epoch <span style="color:#008c00; ">96</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1563</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1563</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">10</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">7</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0284</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9921</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9971</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8495</span>
Epoch <span style="color:#008c00; ">97</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1563</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1563</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">10</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">7</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0233</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9937</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">1.1636</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8468</span>
Epoch <span style="color:#008c00; ">98</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1563</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1563</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">10</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">7</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0276</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9920</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">1.0155</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8474</span>
Epoch <span style="color:#008c00; ">99</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1563</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1563</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">10</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">7</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0303</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9913</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">1.1582</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8414</span>
Epoch <span style="color:#008c00; ">100</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1563</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1563</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">10</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">7</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0273</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9922</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">1.1399</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8398</span>
</pre>

### 4.6. Part 6: Evaluate the model

* Evaluate the trained CNN model on the test data using different evaluation metrics:
  * Loss function
  * Accuracy
  * Confusion matrix.

### 4.6.1. Loss function:

* Display the variations of the training and validation loss function with the number of epochs:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Plot loss per iteration</span>
<span style="color:#800000; font-weight:bold; ">import</span> matplotlib<span style="color:#808030; ">.</span>pyplot <span style="color:#800000; font-weight:bold; ">as</span> plt
plt<span style="color:#808030; ">.</span>plot<span style="color:#808030; ">(</span>result<span style="color:#808030; ">.</span>history<span style="color:#808030; ">[</span><span style="color:#0000e6; ">'loss'</span><span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> label<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'loss'</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>plot<span style="color:#808030; ">(</span>result<span style="color:#808030; ">.</span>history<span style="color:#808030; ">[</span><span style="color:#0000e6; ">'val_loss'</span><span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> label<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'val_loss'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>legend<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>xlabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Epoch Iteration'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>ylabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Loss'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
</pre>

<img width="500" src="images/loss-function-before-augmentation.png">

#### 4.6.3. Accuracy:

* Display the variations of the training and validation accuracy with the number of epochs:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Plot accuracy per iteration</span>
plt<span style="color:#808030; ">.</span>plot<span style="color:#808030; ">(</span>result<span style="color:#808030; ">.</span>history<span style="color:#808030; ">[</span><span style="color:#0000e6; ">'accuracy'</span><span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> label<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'acc'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>plot<span style="color:#808030; ">(</span>result<span style="color:#808030; ">.</span>history<span style="color:#808030; ">[</span><span style="color:#0000e6; ">'val_accuracy'</span><span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> label<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'val_acc'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>legend<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>xlabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Epoch Iteration'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>ylabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Accuracy'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
</pre>

<img width="500" src="images/accuracy-before-augmentation.png">


#### 4.6.3. Compute the test-data Accuracy:

* Compute and display the accuracy on the test-data:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Compute the model accuracy on the test data</span>
accuracy_test_data <span style="color:#808030; ">=</span> model<span style="color:#808030; ">.</span>evaluate<span style="color:#808030; ">(</span>x_test<span style="color:#808030; ">,</span> y_test<span style="color:#808030; ">)</span>
<span style="color:#696969; "># display the atest-data accuracy</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'-------------------------------------------------------'</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'The test-data accuracy = '</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#808030; ">(</span>accuracy_test_data<span style="color:#808030; ">[</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'-------------------------------------------------------'</span><span style="color:#808030; ">)</span>

<span style="color:#008c00; ">313</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">313</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">1</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">3</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">1.1399</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8398</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
The test<span style="color:#44aadd; ">-</span>data accuracy <span style="color:#808030; ">=</span> <span style="color:#008000; ">0.8398000001907349</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
</pre>


#### 4.6.4. Confusion Matrix Visualizations:

* Compute the confusion matrix:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Compute the confusion matrix</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#800000; font-weight:bold; ">def</span> plot_confusion_matrix<span style="color:#808030; ">(</span>cm<span style="color:#808030; ">,</span> classes<span style="color:#808030; ">,</span>
                          normalize<span style="color:#808030; ">=</span><span style="color:#074726; ">False</span><span style="color:#808030; ">,</span>
                          title<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'Confusion matrix'</span><span style="color:#808030; ">,</span>
                          cmap<span style="color:#808030; ">=</span>plt<span style="color:#808030; ">.</span>cm<span style="color:#808030; ">.</span>Blues<span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
  <span style="color:#696969; ">"""</span>
<span style="color:#696969; ">&nbsp;&nbsp;This function prints and plots the confusion matrix.</span>
<span style="color:#696969; ">&nbsp;&nbsp;Normalization can be applied by setting `normalize=True`.</span>
<span style="color:#696969; ">&nbsp;&nbsp;"""</span>
  <span style="color:#800000; font-weight:bold; ">if</span> normalize<span style="color:#808030; ">:</span>
      cm <span style="color:#808030; ">=</span> cm<span style="color:#808030; ">.</span>astype<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'float'</span><span style="color:#808030; ">)</span> <span style="color:#44aadd; ">/</span> cm<span style="color:#808030; ">.</span><span style="color:#400000; ">sum</span><span style="color:#808030; ">(</span>axis<span style="color:#808030; ">=</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span><span style="color:#808030; ">[</span><span style="color:#808030; ">:</span><span style="color:#808030; ">,</span> np<span style="color:#808030; ">.</span>newaxis<span style="color:#808030; ">]</span>
      <span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Normalized confusion matrix"</span><span style="color:#808030; ">)</span>
  <span style="color:#800000; font-weight:bold; ">else</span><span style="color:#808030; ">:</span>
      <span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Confusion matrix, without normalization'</span><span style="color:#808030; ">)</span>

  <span style="color:#696969; "># Display the confusuon matrix</span>
  <span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span>cm<span style="color:#808030; ">)</span>
  <span style="color:#696969; "># display the confusion matrix</span>
  plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>cm<span style="color:#808030; ">,</span> interpolation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'nearest'</span><span style="color:#808030; ">,</span> cmap<span style="color:#808030; ">=</span>cmap<span style="color:#808030; ">)</span>
  plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span>title<span style="color:#808030; ">)</span>
  plt<span style="color:#808030; ">.</span>colorbar<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
  tick_marks <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>arange<span style="color:#808030; ">(</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>classes<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
  plt<span style="color:#808030; ">.</span>xticks<span style="color:#808030; ">(</span>tick_marks<span style="color:#808030; ">,</span> classes<span style="color:#808030; ">,</span> rotation<span style="color:#808030; ">=</span><span style="color:#008c00; ">45</span><span style="color:#808030; ">)</span>
  plt<span style="color:#808030; ">.</span>yticks<span style="color:#808030; ">(</span>tick_marks<span style="color:#808030; ">,</span> classes<span style="color:#808030; ">)</span>
  
  fmt <span style="color:#808030; ">=</span> <span style="color:#0000e6; ">'.2f'</span> <span style="color:#800000; font-weight:bold; ">if</span> normalize <span style="color:#800000; font-weight:bold; ">else</span> <span style="color:#0000e6; ">'d'</span>
  thresh <span style="color:#808030; ">=</span> cm<span style="color:#808030; ">.</span><span style="color:#400000; ">max</span><span style="color:#808030; ">(</span><span style="color:#808030; ">)</span> <span style="color:#44aadd; ">/</span> <span style="color:#008000; ">2.</span>
  <span style="color:#800000; font-weight:bold; ">for</span> i<span style="color:#808030; ">,</span> j <span style="color:#800000; font-weight:bold; ">in</span> itertools<span style="color:#808030; ">.</span>product<span style="color:#808030; ">(</span><span style="color:#400000; ">range</span><span style="color:#808030; ">(</span>cm<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span>cm<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
      plt<span style="color:#808030; ">.</span>text<span style="color:#808030; ">(</span>j<span style="color:#808030; ">,</span> i<span style="color:#808030; ">,</span> format<span style="color:#808030; ">(</span>cm<span style="color:#808030; ">[</span>i<span style="color:#808030; ">,</span> j<span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> fmt<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span>
               horizontalalignment<span style="color:#808030; ">=</span><span style="color:#0000e6; ">"center"</span><span style="color:#808030; ">,</span>
               color<span style="color:#808030; ">=</span><span style="color:#0000e6; ">"white"</span> <span style="color:#800000; font-weight:bold; ">if</span> cm<span style="color:#808030; ">[</span>i<span style="color:#808030; ">,</span> j<span style="color:#808030; ">]</span> <span style="color:#44aadd; ">&gt;</span> thresh <span style="color:#800000; font-weight:bold; ">else</span> <span style="color:#0000e6; ">"black"</span><span style="color:#808030; ">)</span>

  plt<span style="color:#808030; ">.</span>tight_layout<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
  plt<span style="color:#808030; ">.</span>ylabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'True label'</span><span style="color:#808030; ">)</span>
  plt<span style="color:#808030; ">.</span>xlabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Predicted label'</span><span style="color:#808030; ">)</span>
  plt<span style="color:#808030; ">.</span>show<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>

<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Predict the targets for the test data</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
p_test <span style="color:#808030; ">=</span> model<span style="color:#808030; ">.</span>predict<span style="color:#808030; ">(</span>x_test<span style="color:#808030; ">)</span><span style="color:#808030; ">.</span>argmax<span style="color:#808030; ">(</span>axis<span style="color:#808030; ">=</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># construct the confusion matrix</span>
cm <span style="color:#808030; ">=</span> confusion_matrix<span style="color:#808030; ">(</span>y_test<span style="color:#808030; ">,</span> p_test<span style="color:#808030; ">)</span>
<span style="color:#696969; "># plot the confusion matrix</span>
plot_confusion_matrix<span style="color:#808030; ">(</span>cm<span style="color:#808030; ">,</span> <span style="color:#400000; ">list</span><span style="color:#808030; ">(</span><span style="color:#400000; ">range</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">10</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> 
                      <span style="color:#074726; ">False</span><span style="color:#808030; ">,</span> 
                      <span style="color:#0000e6; ">'Confusion matrix'</span><span style="color:#808030; ">,</span> 
                      plt<span style="color:#808030; ">.</span>cm<span style="color:#808030; ">.</span>Greens<span style="color:#808030; ">)</span>
</pre>


<img width="500" src="images/confusion-matrix-before-augmentation.png">

#### 4.6.5) Examine some of the misclassified digits:

* Display some of the misclassified digit:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># - Find the indices of all the mis-classified examples</span>
misclassified_idx <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>where<span style="color:#808030; ">(</span>p_test <span style="color:#44aadd; ">!=</span> y_test<span style="color:#808030; ">)</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span> <span style="color:#696969; "># select the index</span>
<span style="color:#696969; "># setup the subplot grid for the visualized images</span>
 <span style="color:#696969; "># the suplot grid shape</span>
num_rows <span style="color:#808030; ">=</span> <span style="color:#008c00; ">5</span>
<span style="color:#696969; "># the number of columns</span>
num_cols <span style="color:#808030; ">=</span> num_visualized_images <span style="color:#44aadd; ">//</span> num_rows
<span style="color:#696969; "># setup the subplots axes</span>
fig<span style="color:#808030; ">,</span> axes <span style="color:#808030; ">=</span> plt<span style="color:#808030; ">.</span>subplots<span style="color:#808030; ">(</span>nrows<span style="color:#808030; ">=</span>num_rows<span style="color:#808030; ">,</span> ncols<span style="color:#808030; ">=</span>num_cols<span style="color:#808030; ">,</span> figsize<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">12</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># set a seed random number generator for reproducible results</span>
seed<span style="color:#808030; ">(</span>random_state_seed<span style="color:#808030; ">)</span>
<span style="color:#696969; "># iterate over the sub-plots</span>
<span style="color:#800000; font-weight:bold; ">for</span> row <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span>num_rows<span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
  <span style="color:#800000; font-weight:bold; ">for</span> col <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span>num_cols<span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
    <span style="color:#696969; "># get the next figure axis</span>
    ax <span style="color:#808030; ">=</span> axes<span style="color:#808030; ">[</span>row<span style="color:#808030; ">,</span> col<span style="color:#808030; ">]</span><span style="color:#808030; ">;</span>
    <span style="color:#696969; "># turn-off subplot axis</span>
    ax<span style="color:#808030; ">.</span>set_axis_off<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># select a random mis-classified example</span>
    counter <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>random<span style="color:#808030; ">.</span>choice<span style="color:#808030; ">(</span>misclassified_idx<span style="color:#808030; ">)</span>
    <span style="color:#696969; "># get test image </span>
    image <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>squeeze<span style="color:#808030; ">(</span>x_test<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">,</span><span style="color:#808030; ">:</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># get the true labels of the selected image</span>
    label <span style="color:#808030; ">=</span> labels<span style="color:#808030; ">[</span>y_test<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">]</span><span style="color:#808030; ">]</span>
    <span style="color:#696969; "># get the predicted label of the test image</span>
    yhat <span style="color:#808030; ">=</span> labels<span style="color:#808030; ">[</span>p_test<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">]</span><span style="color:#808030; ">]</span>
    <span style="color:#696969; "># display the image </span>
    ax<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>image<span style="color:#808030; ">,</span> cmap<span style="color:#808030; ">=</span>plt<span style="color:#808030; ">.</span>cm<span style="color:#808030; ">.</span>gray_r<span style="color:#808030; ">,</span> interpolation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'nearest'</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># display the true and predicted labels on the title of tehe image</span>
    ax<span style="color:#808030; ">.</span>set_title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Y = %s, $</span><span style="color:#0f69ff; ">\h</span><span style="color:#0000e6; ">at{Y}$ = %s'</span> <span style="color:#44aadd; ">%</span> <span style="color:#808030; ">(</span><span style="color:#808030; ">(</span>label<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span>yhat<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> size <span style="color:#808030; ">=</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">)</span>
</pre>


<img width="500" src="images/25-misclassifications-before-augmentation.png">

### 4.7. Part 7: Try to improve the model performance using data augmentation:

* The CNN appears to be over-fitting:
  * Excellent performance on the training data
  * Poor performance on the validation data.

* In an attempt to address the over-fitting issue, we apply data augmentation:

  * Data augmentation simply means increasing size of the labelled data so that we provide higher number of training and validation examples:
  * Some of the popular image augmentation techniques are flipping, translation, rotation, scaling, changing brightness, adding noise, etc.
  * Next, we apply data augmentation using the folloing image transformation:
     * Horizontal image shift by 10% or less
     * Vertical image shift by 10% or less
     * Horizontal image flip.


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># 7) Apply data augmentation:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Set the batch size</span>
batch_size <span style="color:#808030; ">=</span> <span style="color:#008c00; ">32</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># 7.1) Define the data generator: defines the way the training images are augmented</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - width_shift_range=0.1</span>
<span style="color:#696969; "># - height_shift_range=0.1</span>
<span style="color:#696969; "># - horizontal_flip=True</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
data_generator <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>preprocessing<span style="color:#808030; ">.</span>image<span style="color:#808030; ">.</span>ImageDataGenerator<span style="color:#808030; ">(</span>width_shift_range<span style="color:#808030; ">=</span><span style="color:#008000; ">0.1</span><span style="color:#808030; ">,</span> height_shift_range<span style="color:#808030; ">=</span><span style="color:#008000; ">0.1</span><span style="color:#808030; ">,</span> horizontal_flip<span style="color:#808030; ">=</span><span style="color:#074726; ">True</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># data generator</span>
train_generator <span style="color:#808030; ">=</span> data_generator<span style="color:#808030; ">.</span>flow<span style="color:#808030; ">(</span>x_train<span style="color:#808030; ">,</span> y_train<span style="color:#808030; ">,</span> batch_size<span style="color:#808030; ">)</span>
<span style="color:#696969; "># compute the steps per epochs</span>
steps_per_epoch <span style="color:#808030; ">=</span> x_train<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">//</span> batch_size

<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># 7.2) Fit the model using the augmented data</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Note: if you run this AFTER calling the previous model.fit(), </span>
<span style="color:#696969; ">#   - It will CONTINUE training where it left off</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - set the number of epochs</span>
num_epochs <span style="color:#808030; ">=</span> <span style="color:#008c00; ">100</span>
<span style="color:#696969; "># train the model</span>
r <span style="color:#808030; ">=</span> model<span style="color:#808030; ">.</span>fit<span style="color:#808030; ">(</span>train_generator<span style="color:#808030; ">,</span> validation_data<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span>x_test<span style="color:#808030; ">,</span> y_test<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> 
              steps_per_epoch<span style="color:#808030; ">=</span>steps_per_epoch<span style="color:#808030; ">,</span> 
              epochs<span style="color:#808030; ">=</span>num_epochs<span style="color:#808030; ">)</span>
</pre>


<pre style="color:#000000;background:#ffffff;">Epoch <span style="color:#008c00; ">1</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1562</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1562</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">30</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">19</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.5678</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8228</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.4802</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8533</span>
Epoch <span style="color:#008c00; ">2</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1562</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1562</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">29</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">19</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.4467</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8572</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.4519</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8543</span>
Epoch <span style="color:#008c00; ">3</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1562</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1562</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">29</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">19</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.4121</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8672</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.4935</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8491</span>
Epoch <span style="color:#008c00; ">4</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1562</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1562</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">29</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">19</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.3962</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8713</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.4293</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8599</span>
Epoch <span style="color:#008c00; ">5</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1562</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1562</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">30</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">19</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.3654</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8788</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.4271</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8633</span>
<span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span>
<span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span>
Epoch <span style="color:#008c00; ">95</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1562</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1562</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">29</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">19</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.1259</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9592</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.4579</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8909</span>
Epoch <span style="color:#008c00; ">96</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1562</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1562</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">29</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">19</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.1273</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9586</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.4829</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8892</span>
Epoch <span style="color:#008c00; ">97</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1562</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1562</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">29</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">19</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.1237</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9596</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.4814</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8857</span>
Epoch <span style="color:#008c00; ">98</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1562</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1562</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">29</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">19</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.1260</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9602</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.4297</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8886</span>
Epoch <span style="color:#008c00; ">99</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1562</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1562</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">30</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">19</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.1239</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9589</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.4830</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8809</span>
Epoch <span style="color:#008c00; ">100</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1562</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1562</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">30</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">19</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.1231</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9607</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.4466</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8912</span>
</pre>


### 4.8. Part 8: Evaluate the re-trained CNN model using data augmentation:

* Evaluate the trained CNN model on the test data using different evaluation metrics:
   * Loss function
   * Accuracy
   * Confusion matrix.

* 4.8.1. Loss function:

  * Display the variations of the training and validation loss function with the number of epochs:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Plot loss per iteration</span>
<span style="color:#800000; font-weight:bold; ">import</span> matplotlib<span style="color:#808030; ">.</span>pyplot <span style="color:#800000; font-weight:bold; ">as</span> plt
plt<span style="color:#808030; ">.</span>plot<span style="color:#808030; ">(</span>r<span style="color:#808030; ">.</span>history<span style="color:#808030; ">[</span><span style="color:#0000e6; ">'loss'</span><span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> label<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'loss'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>plot<span style="color:#808030; ">(</span>r<span style="color:#808030; ">.</span>history<span style="color:#808030; ">[</span><span style="color:#0000e6; ">'val_loss'</span><span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> label<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'val_loss'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>legend<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>xlabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Iteration number'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>ylabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Loss'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Loss as a Function of the Iteration Number'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
</pre>

<img width="500" src="images/loss-function-after-augmentation.png">


#### 4.8.2. Accuracy:

* Display the variations of the training and validation accuracy with the number of epochs:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Plot accuracy per iteration</span>
plt<span style="color:#808030; ">.</span>plot<span style="color:#808030; ">(</span>r<span style="color:#808030; ">.</span>history<span style="color:#808030; ">[</span><span style="color:#0000e6; ">'accuracy'</span><span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> label<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'acc'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>plot<span style="color:#808030; ">(</span>r<span style="color:#808030; ">.</span>history<span style="color:#808030; ">[</span><span style="color:#0000e6; ">'val_accuracy'</span><span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> label<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'val_acc'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>legend<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>xlabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Iteration number'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>ylabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Accuracy'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Accuracy as a Function of the Iteration Number'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
</pre>

<img width="500" src="images/accuracy-after-augmentation.png">

#### 4.8.3. Compute the test-data Accuracy:

* Compute and display the accuracy on the test-data:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Compute the model accuracy on the test data</span>
accuracy_test_data <span style="color:#808030; ">=</span> model<span style="color:#808030; ">.</span>evaluate<span style="color:#808030; ">(</span>x_test<span style="color:#808030; ">,</span> y_test<span style="color:#808030; ">)</span>
<span style="color:#696969; "># display the atest-data accuracy</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'-------------------------------------------------------'</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'The test-data accuracy = '</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#808030; ">(</span>accuracy_test_data<span style="color:#808030; ">[</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'-------------------------------------------------------'</span><span style="color:#808030; ">)</span>
</pre>



<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#008c00; ">313</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">313</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">1</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">3</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.4466</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8912</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
The test<span style="color:#44aadd; ">-</span>data accuracy <span style="color:#808030; ">=</span> <span style="color:#008000; ">0.8912000060081482</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
</pre>


#### 4.8.4. Confusion Matrix Visualizations:

* Compute the confusion matrix:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Plot confusion matrix</span>
<span style="color:#800000; font-weight:bold; ">from</span> sklearn<span style="color:#808030; ">.</span>metrics <span style="color:#800000; font-weight:bold; ">import</span> confusion_matrix
<span style="color:#800000; font-weight:bold; ">import</span> itertools

<span style="color:#800000; font-weight:bold; ">def</span> plot_confusion_matrix<span style="color:#808030; ">(</span>cm<span style="color:#808030; ">,</span> classes<span style="color:#808030; ">,</span>
                          normalize<span style="color:#808030; ">=</span><span style="color:#074726; ">False</span><span style="color:#808030; ">,</span>
                          title<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'Confusion matrix'</span><span style="color:#808030; ">,</span>
                          cmap<span style="color:#808030; ">=</span>plt<span style="color:#808030; ">.</span>cm<span style="color:#808030; ">.</span>Greens<span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
  <span style="color:#696969; ">"""</span>
<span style="color:#696969; ">&nbsp;&nbsp;This function prints and plots the confusion matrix.</span>
<span style="color:#696969; ">&nbsp;&nbsp;Normalization can be applied by setting `normalize=True`.</span>
<span style="color:#696969; ">&nbsp;&nbsp;"""</span>
  <span style="color:#800000; font-weight:bold; ">if</span> normalize<span style="color:#808030; ">:</span>
      cm <span style="color:#808030; ">=</span> cm<span style="color:#808030; ">.</span>astype<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'float'</span><span style="color:#808030; ">)</span> <span style="color:#44aadd; ">/</span> cm<span style="color:#808030; ">.</span><span style="color:#400000; ">sum</span><span style="color:#808030; ">(</span>axis<span style="color:#808030; ">=</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span><span style="color:#808030; ">[</span><span style="color:#808030; ">:</span><span style="color:#808030; ">,</span> np<span style="color:#808030; ">.</span>newaxis<span style="color:#808030; ">]</span>
      <span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Normalized confusion matrix"</span><span style="color:#808030; ">)</span>
  <span style="color:#800000; font-weight:bold; ">else</span><span style="color:#808030; ">:</span>
      <span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Confusion matrix, without normalization'</span><span style="color:#808030; ">)</span>

  <span style="color:#696969; "># display the confusion matrix</span>
  <span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span>cm<span style="color:#808030; ">)</span>

  plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>cm<span style="color:#808030; ">,</span> interpolation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'nearest'</span><span style="color:#808030; ">,</span> cmap<span style="color:#808030; ">=</span>cmap<span style="color:#808030; ">)</span>
  plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span>title<span style="color:#808030; ">)</span>
  plt<span style="color:#808030; ">.</span>colorbar<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
  tick_marks <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>arange<span style="color:#808030; ">(</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>classes<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
  plt<span style="color:#808030; ">.</span>xticks<span style="color:#808030; ">(</span>tick_marks<span style="color:#808030; ">,</span> classes<span style="color:#808030; ">,</span> rotation<span style="color:#808030; ">=</span><span style="color:#008c00; ">45</span><span style="color:#808030; ">)</span>
  plt<span style="color:#808030; ">.</span>yticks<span style="color:#808030; ">(</span>tick_marks<span style="color:#808030; ">,</span> classes<span style="color:#808030; ">)</span>

  fmt <span style="color:#808030; ">=</span> <span style="color:#0000e6; ">'.2f'</span> <span style="color:#800000; font-weight:bold; ">if</span> normalize <span style="color:#800000; font-weight:bold; ">else</span> <span style="color:#0000e6; ">'d'</span>
  thresh <span style="color:#808030; ">=</span> cm<span style="color:#808030; ">.</span><span style="color:#400000; ">max</span><span style="color:#808030; ">(</span><span style="color:#808030; ">)</span> <span style="color:#44aadd; ">/</span> <span style="color:#008000; ">2.</span>
  <span style="color:#800000; font-weight:bold; ">for</span> i<span style="color:#808030; ">,</span> j <span style="color:#800000; font-weight:bold; ">in</span> itertools<span style="color:#808030; ">.</span>product<span style="color:#808030; ">(</span><span style="color:#400000; ">range</span><span style="color:#808030; ">(</span>cm<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span>cm<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
      plt<span style="color:#808030; ">.</span>text<span style="color:#808030; ">(</span>j<span style="color:#808030; ">,</span> i<span style="color:#808030; ">,</span> format<span style="color:#808030; ">(</span>cm<span style="color:#808030; ">[</span>i<span style="color:#808030; ">,</span> j<span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> fmt<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span>
               horizontalalignment<span style="color:#808030; ">=</span><span style="color:#0000e6; ">"center"</span><span style="color:#808030; ">,</span>
               color<span style="color:#808030; ">=</span><span style="color:#0000e6; ">"white"</span> <span style="color:#800000; font-weight:bold; ">if</span> cm<span style="color:#808030; ">[</span>i<span style="color:#808030; ">,</span> j<span style="color:#808030; ">]</span> <span style="color:#44aadd; ">&gt;</span> thresh <span style="color:#800000; font-weight:bold; ">else</span> <span style="color:#0000e6; ">"black"</span><span style="color:#808030; ">)</span>

  plt<span style="color:#808030; ">.</span>tight_layout<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
  plt<span style="color:#808030; ">.</span>ylabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'True label'</span><span style="color:#808030; ">)</span>
  plt<span style="color:#808030; ">.</span>xlabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Predicted label'</span><span style="color:#808030; ">)</span>
  plt<span style="color:#808030; ">.</span>show<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>


<span style="color:#696969; "># prediction </span>
p_test <span style="color:#808030; ">=</span> model<span style="color:#808030; ">.</span>predict<span style="color:#808030; ">(</span>x_test<span style="color:#808030; ">)</span><span style="color:#808030; ">.</span>argmax<span style="color:#808030; ">(</span>axis<span style="color:#808030; ">=</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># compute confusuon matrix</span>
cm <span style="color:#808030; ">=</span> confusion_matrix<span style="color:#808030; ">(</span>y_test<span style="color:#808030; ">,</span> p_test<span style="color:#808030; ">)</span>
<span style="color:#696969; "># visualie the confusion matrix</span>
plot_confusion_matrix<span style="color:#808030; ">(</span>cm<span style="color:#808030; ">,</span> <span style="color:#400000; ">list</span><span style="color:#808030; ">(</span><span style="color:#400000; ">range</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">10</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
</pre>


<img width="500" src="images/confusion-matrix-after-augmentation.png">



#### 4.8.5. Examine some of the misclassified test-data examples:

* Display some of the misclassified items from the test data:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># - Find the indices of all the mis-classified examples</span>
misclassified_idx <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>where<span style="color:#808030; ">(</span>p_test <span style="color:#44aadd; ">!=</span> y_test<span style="color:#808030; ">)</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span> <span style="color:#696969; "># select the index</span>
<span style="color:#696969; "># setup the subplot grid for the visualized images</span>
 <span style="color:#696969; "># the suplot grid shape</span>
num_rows <span style="color:#808030; ">=</span> <span style="color:#008c00; ">5</span>
<span style="color:#696969; "># the number of columns</span>
num_cols <span style="color:#808030; ">=</span> num_visualized_images <span style="color:#44aadd; ">//</span> num_rows
<span style="color:#696969; "># setup the subplots axes</span>
fig<span style="color:#808030; ">,</span> axes <span style="color:#808030; ">=</span> plt<span style="color:#808030; ">.</span>subplots<span style="color:#808030; ">(</span>nrows<span style="color:#808030; ">=</span>num_rows<span style="color:#808030; ">,</span> ncols<span style="color:#808030; ">=</span>num_cols<span style="color:#808030; ">,</span> figsize<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">12</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># set a seed random number generator for reproducible results</span>
seed<span style="color:#808030; ">(</span>random_state_seed<span style="color:#808030; ">)</span>
<span style="color:#696969; "># iterate over the sub-plots</span>
<span style="color:#800000; font-weight:bold; ">for</span> row <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span>num_rows<span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
  <span style="color:#800000; font-weight:bold; ">for</span> col <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span>num_cols<span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
    <span style="color:#696969; "># get the next figure axis</span>
    ax <span style="color:#808030; ">=</span> axes<span style="color:#808030; ">[</span>row<span style="color:#808030; ">,</span> col<span style="color:#808030; ">]</span><span style="color:#808030; ">;</span>
    <span style="color:#696969; "># turn-off subplot axis</span>
    ax<span style="color:#808030; ">.</span>set_axis_off<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># select a random mis-classified example</span>
    counter <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>random<span style="color:#808030; ">.</span>choice<span style="color:#808030; ">(</span>misclassified_idx<span style="color:#808030; ">)</span>
    <span style="color:#696969; "># get test image </span>
    image <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>squeeze<span style="color:#808030; ">(</span>x_test<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">,</span><span style="color:#808030; ">:</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># get the true labels of the selected image</span>
    label <span style="color:#808030; ">=</span> labels<span style="color:#808030; ">[</span>y_test<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">]</span><span style="color:#808030; ">]</span>
    <span style="color:#696969; "># get the predicted label of the test image</span>
    yhat <span style="color:#808030; ">=</span> labels<span style="color:#808030; ">[</span>p_test<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">]</span><span style="color:#808030; ">]</span>
    <span style="color:#696969; "># display the image </span>
    ax<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>image<span style="color:#808030; ">,</span> cmap<span style="color:#808030; ">=</span>plt<span style="color:#808030; ">.</span>cm<span style="color:#808030; ">.</span>gray_r<span style="color:#808030; ">,</span> interpolation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'nearest'</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># display the true and predicted labels on the title of tehe image</span>
    ax<span style="color:#808030; ">.</span>set_title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Y = %s, $</span><span style="color:#0f69ff; ">\h</span><span style="color:#0000e6; ">at{Y}$ = %s'</span> <span style="color:#44aadd; ">%</span> <span style="color:#808030; ">(</span><span style="color:#808030; ">(</span>label<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span>yhat<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> size <span style="color:#808030; ">=</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">)</span>
</pre>

<img width="500" src="images/25-misclassifications-after-augmentation.png">

### 4.9. Part 9: Display a final message after successful execution completion

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># display a final message</span>
<span style="color:#696969; "># current time</span>
now <span style="color:#808030; ">=</span> datetime<span style="color:#808030; ">.</span>datetime<span style="color:#808030; ">.</span>now<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># display a message</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Program executed successfully on: '</span><span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#808030; ">(</span>now<span style="color:#808030; ">.</span>strftime<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"%Y-%m-%d %H:%M:%S"</span><span style="color:#808030; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#0000e6; ">"...Goodbye!</span><span style="color:#0f69ff; ">\n</span><span style="color:#0000e6; ">"</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>

Program executed successfully on<span style="color:#808030; ">:</span> <span style="color:#008c00; ">2021</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">04</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">05</span> <span style="color:#008c00; ">19</span><span style="color:#808030; ">:</span><span style="color:#008c00; ">42</span><span style="color:#808030; ">:</span><span style="color:#008000; ">07.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span>Goodbye!
</pre>


## 5. Analysis

* In view of the presented results, we make the following observations:

   * Before augmentation original designed CNN did not achieves reasonably good classification accuracy on CIFAR-10 dataset:
   * Achieved classification accuracy on the test data = 84%
   * The CNN appears to be over-fitting:
   * Excellent performance on the training data
   * Poor performance on the validation data
   * In an attempt to address the over-fitting issue, we apply data augmentation:
     * Data augmentation simply means increasing size of the labelled data so that we provide higher number of training and validation examples
     * Some of the popular image augmentation techniques are flipping, translation, rotation, scaling, changing brightness, adding noise, etc.
     * We applied data augmentation using the following image transformation:
     * Horizontal image shift by 10% or less
     * Vertical image shift by 10% or less
     * Horizontal image flip.

  * After augmentation the designed CNN achieved much better classification accuracy on CIFAR-10 dataset:
  * Achieved classification accuracy on the test data = 84%

### 4.6. Future Work

* We plan to explore the following related issues:

    * To explore ways of improving the performance of this simple CNN, including fine-tuning the following hyper-parameters:
    * Even after data augmentation, the validation loss function is increasing while the training loss function is decreasing:
      * This indicates over-fitting
      * We shall address this over-fitting by adding dropout and batch normalization layers.
      * We shall also explore fine-tuning some of the hyper-parameters, including:
        * The number of filters and layers
        * The dropout rate
        * The optimizer
        * The learning rate.

### 4.7. References

1. The CIFAR-10 dataset. Retrieved from: https://www.cs.toronto.edu/~kriz/cifar.html (April 6th, 2021). 
2. Park Chansung. CIFAR-10 Image Classification in TensorFlow. Retrieved from: https://towardsdatascience.com/cifar-10-image-classification-in-tensorflow-5b501f7dc77c (April 6th, 2021).
3. Jason Brownlee. How to Develop a CNN From Scratch for CIFAR-10 Photo Classification. Retrieved from: https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification (April 6th, 2021).
4. Tensorflow. Convolutional Neural Network (CNN). Retrieved from: https://www.tensorflow.org/tutorials/images/cnn (April 6th, 2021).
5. Aarya Brahmane.  Deep Learning with CIFAR-10:  Image Classification using CNN. Retrieved from: https://towardsdatascience.com/ (April 6th, 2021).
6. Abhijeet Kumar. Achieving 90% accuracy in Object Recognition Task on CIFAR- Dataset with Keras: Convolutional Neural Networks. Retrieved from: Retrieved from: https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/ (April 6th, 2021).
7. Konstantinos Siaterlis. Convolutional NN with Keras Tensorflow on CIFAR-10 Dataset, Image Classification. Retrieved from: https://medium.com/@siakon/convolutional-nn-with-keras-tensorflow-on-cifar-10-dataset-image-classification-d3aad44691bd (April 6th, 2021).
8. Mia Morton. Experimental Process: Completing a Convolutional Neural Network to Classify the CIFAR 10 Dataset. Retrieved from: https://medium.com/@704/experimental-process-in-completing-convolutional-neural-network-to-classify-the-cifar-10-dataset-8de699b82b8d (April 6th, 2021).
9. Kaggle. CIFAR-10 - Object Recognition in Images
Identify the subject of 60,000 labeled images. Retrieved from: https://www.kaggle.com/c/cifar-10/discussion/40237 (April 6th, 2021).
10. Aarya Brahmane. Deep Learning with CIFAR-10. Retrieved from: https://towardsdatascience.com/deep-learning-with-cifar-10-image-classification-64ab92110d79 (April 6th, 2021).
Tensorflow. Convolutional Neural Network (CNN). Retrieved from: https://www.tensorflow.org/tutorials/images/cnn (April 6th, 2021).


