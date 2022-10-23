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
