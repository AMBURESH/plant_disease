# plant_disease
A deep learning-based application for detecting plant diseases from leaf .This project implements a Convolutional Neural Network (CNN) using PyTorch to classify plant leaf images into 38 different disease categories. The model is trained on a custom dataset and leverages image preprocessing, CNN layers, and evaluation metrics like accuracy and classification report.

MODEL OVERVIEW

.Framework: PyTorch

.Architecture: Custom CNN with 3 convolutional layers, max & avg pooling, and 5 fully connected layers

.Output Classes: 38

.Loss Function: CrossEntropyLoss

.Optimizer: Adam

.Scheduler: StepLR (decays LR every 10 epochs)
