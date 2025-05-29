# Neural Networks from Scratch – Implementation with NumPy

This project implements a complete **neural network framework from scratch using NumPy**, following a layered architecture approach. The goal is to understand and implement the core components of deep learning pipelines, including forward and backward passes, loss functions, optimizers, and training routines, without using deep learning libraries like TensorFlow or PyTorch.

## Objectives

- Implement fundamental layers and activations: Fully Connected, ReLU, Sigmoid, Softmax.
- Implement loss functions: MSE and Softmax Cross Entropy.
- Implement training using gradient descent and momentum.
- Train the network on classification (MNIST) and regression (California Housing) tasks.

## Components Implemented

- **Layers**:
  - FullyConnected (Affine)
  - ReLU
  - Sigmoid
  - Composed (e.g., FullyConnected + ReLU)

- **Loss Functions**:
  - Mean Squared Error (MSE)
  - Softmax with Cross Entropy

- **Forward & Backward Passes**:
  - Each layer supports gradient computation
  - Layer gradients verified with numerical checking

- **Optimizer**:
  - SGD with Momentum
  - Custom `sgd_momentum` implemented from scratch

- **Batch Normalization**:
  - Optional layer for improving training convergence

## Training and Evaluation

- **Datasets**:
  - MNIST for classification of handwritten digits
  - California Housing dataset for regression tasks

- **Training Details**:
  - Manual implementation of epoch loop and parameter updates
  - Accuracy and loss tracked over iterations
  - Early stopping and batch updates supported

## Key Learnings

- Backpropagation implementation improves understanding of weight updates.
- Building the network from zero clarifies the function of each layer and optimizer.
- Layer composability and debugging becomes easier with modular design.

> Course: Artificial Intelligence – Fall 1403  
> University of Tehran | School of Electrical & Computer Engineering