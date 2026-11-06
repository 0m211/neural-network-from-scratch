# Neural Network From Scratch Using NumPy

## Overview

An educational deep learning framework built from scratch using Python and NumPy. The project implements core neural network components and evaluates their behavior through controlled experiments.

## Key Features

* Custom automatic differentiation engine
* Fully connected neural network layers
* Activation functions and loss functions
* Gradient-based optimization
* Dropout regularization
* Batch normalization
* Model saving and loading
* Numerical gradient checking
* Early stopping and checkpoint selection
* Repeated-seed experiments
* Classification metrics and confusion-matrix analysis

## MNIST Classification Results

**Selected model:** Fully connected neural network with dropout probability 0.2.

* Test accuracy: 97.92%
* Test cross-entropy loss: 0.06553
* Correct predictions: 9,792 out of 10,000
* Macro F1-score: 97.91%

The model was selected using validation data and evaluated on the official MNIST test set.

## Engineering Focus

The project emphasizes understanding neural network internals, validating gradient calculations, comparing optimization and regularization techniques, analyzing model errors, and evaluating predictive confidence.

Additional experiments investigate probability calibration and compare the custom NumPy implementation against PyTorch.

## NumPy vs PyTorch Benchmark

A multi-trial CPU benchmark was conducted to compare the custom NumPy framework against PyTorch using MNIST.

### Experimental Setup

| Parameter | Configuration |
|---|---|
| Dataset | MNIST |
| Architecture | 784 → 128 → 10 |
| Epochs | 3 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Trials | 3 |
| Device | CPU |
| CPU Threads | 1 |
| Environment | WSL2 Ubuntu |

Both implementations used identical initial weights and shared training shuffle orders.

### Benchmark Results

| Metric | NumPy Framework | PyTorch |
|---|---:|---:|
| Training Time | 22.86 ± 3.53 s | 10.75 ± 0.33 s |
| Test Accuracy | 97.20 ± 0.05% | 97.18 ± 0.03% |
| Inference Time | 11.067 ± 0.747 ms | 5.131 ± 0.368 ms |

### Analysis

- PyTorch achieved 2.13× faster training.
- PyTorch achieved 2.16× faster inference.
- The custom NumPy framework achieved comparable classification accuracy.
- Both implementations were evaluated using single-threaded CPU execution.

### Key Takeaway

This experiment demonstrates that a neural network framework built from scratch using NumPy can achieve accuracy comparable to PyTorch while implementing its own automatic differentiation, backpropagation, optimization, and inference pipeline.

*Note: Results represent three trials in a specific CPU/WSL2 environment and are not universal performance measurements.*

## Probability Calibration: Temperature Scaling

Temperature scaling was evaluated to improve the confidence calibration of the selected MNIST model.

The optimal temperature was selected using the validation set and evaluated on the official MNIST test set.

### Results

**Optimal temperature: 1.190**

| Metric | Before Scaling | After Scaling |
|---|---:|---:|
| Test Accuracy | 97.92% | 97.92% |
| Cross-Entropy Loss | 0.06553 | 0.06444 |
| Expected Calibration Error (ECE) | 0.00457 | 0.00433 |
| Brier Score | 0.03044 | 0.03031 |

### Observations

- Test accuracy remained unchanged at 97.92%.
- Cross-entropy loss decreased after calibration.
- ECE and Brier score both showed small improvements.
- Temperature scaling improved probability calibration without changing predicted classes.

The experiment demonstrates how post-training calibration can improve the reliability of model confidence estimates.

**Visualization:** `artifacts/temperature_scaling_comparison.png`
