# Overview
SAIML - Stymphalian AI Machine learning Library

My personal library implementation of common ML/AI data structures, algorithms
and other related functions.
Mainly used as a playground for testing my understanding of the concepts. 
Implemented directly using python and numpy. I will use pytorch and other 
implementation as testing references.

## Features

Feature | Description
--------|------------
Layer | Abstract interface class of a layer of the network. All children will support forward and backward passes
DenseLayer | Fully connected layer
Conv2DLayer | Convolution layer supporting stride, padding, kernel_size and kernel_depth
Pool2DLayer | Pooling layer supporting stride.
Transformer | Basic encoder and decoder blocks of the transformer
Loss | cross_entropy, mean_square_error
PCA | Principle Component Analysis for dimension reduction and feature analysis. Unoptimized and directly computes the eigen values/vectors from the covariance matrix.
autograd | implement autograd tensors (pytorch like api)
GPU | Run all the calculation on GPU (uses cupy)
Optimizer | A general class for running an optimizer for gradient descent (SGD, RMSProp, Adam)

## TODO
Feature  | Description
---|---
Trainer | A class for running the training loop
Diffusion | Basics of diffusion image generation
RNN/LSTM | Basic sequence generation.
Regularization | regularization l1,l2, etc
