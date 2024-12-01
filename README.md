# MNIST Neural Network Implementation

![Python ML Tests](https://github.com/Amlan66/MNISTNeuralNetwork/actions/workflows/python-app.yml/badge.svg)

A PyTorch implementation of a Convolutional Neural Network (CNN) for MNIST digit classification, optimized for high accuracy while maintaining a low parameter count.

## Model Architecture

The model uses a custom CNN architecture with the following key features:

### Network Structure
- **Input**: 28x28 grayscale images
- **3 Main Blocks**:
  1. First Block:
     - Conv2d(1→10, 3x3) + BatchNorm + ReLU
     - Conv2d(10→20, 3x3) + BatchNorm + ReLU
     - MaxPool2d(2x2)
     - Dropout(0.1)
     - Conv1x1(20→10)
  
  2. Second Block:
     - Conv2d(10→20, 3x3) + BatchNorm + ReLU
     - Conv2d(20→20, 3x3) + BatchNorm + ReLU
     - MaxPool2d(2x2)
     - Dropout(0.1)
  
  3. Final Block:
     - Conv2d(20→40, 3x3) + BatchNorm + ReLU
     - Global Average Pooling
     - Conv1x1(40→10)
- **Output**: 10 classes (digits 0-9)

### Key Features
- Total Parameters: ~15,440 (under 20K constraint)
- Batch Normalization for stable training
- Dropout (0.1) for regularization
- Global Average Pooling instead of Dense layers
- 1x1 Convolutions for channel reduction

## Training Details

### Hyperparameters
- Batch Size: 32
- Optimizer: Adam (lr=0.001, weight_decay=1e-4)
- Learning Rate Scheduler: OneCycleLR
  - max_lr: 0.01
  - epochs: 20
  - pct_start: 0.2
  - div_factor: 10.0
  - final_div_factor: 100

### Dataset
- Training: 60,000 MNIST images
- Testing: 10,000 MNIST images
- Normalization: mean=0.1307, std=0.3081

## Tests

The project includes automated tests that verify:
1. Parameter count is under 20,000
2. Proper use of Batch Normalization
3. Implementation of Dropout
4. Use of Global Average Pooling
5. Basic forward pass functionality

The logs for seeing the test accuracy being more than 99.4% are as follows:


![test_accuracy](https://github.com/user-attachments/assets/aebcfc5b-8c92-4036-9187-71ed2a35f2f8)
