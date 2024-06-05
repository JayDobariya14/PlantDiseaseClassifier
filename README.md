# PlantDiseaseClassifier

A deep learning-based project for classifying plant diseases using convolutional neural networks (CNNs). This repository includes code for data processing, model training, and evaluation.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
PlantDiseaseClassifier is designed to classify plant diseases from images. It uses a convolutional neural network (CNN) to learn and predict diseases in various plants, improving agricultural practices by enabling early and accurate disease detection.

## Dataset
The dataset used in this project consists of images of leaves categorized into different disease classes. In this dataset, 39 different classes of plant leaf images are available, containing a total of 61,486 images. For training the model, we randomly choose 40 images (you can change the number in the code given in data folder) from each class, making the training set balanced and manageable. There is a total of 39 classes that we aim to predict using the CNN model.

You can download the dataset from the following link:
[Dataset](https://data.mendeley.com/datasets/tywbtsjrjv/1)

## Model Architecture
The model is built using PyTorch and consists of multiple convolutional layers followed by fully connected layers. Batch normalization and dropout are used to improve the performance and generalization of the model.

## Installation
To run this project, ensure you have Python 3.8+ and the following libraries installed:

- PyTorch
- torchvision
- scikit-learn
- matplotlib
- numpy
- pandas
- jupyter

Install the dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
Data Preprocessing: Process the dataset using the provided Jupyter notebook.
Model Training: Train the model using the training script.
Evaluation: Evaluate the model performance on the test set.

## Results
After training the model, the performance metrics and results will be displayed, including the training and validation accuracy, as well as test accuracy.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any feature requests, bug fixes, or improvements.