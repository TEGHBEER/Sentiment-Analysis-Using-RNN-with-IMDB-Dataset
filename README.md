# Sentiment Analysis Using Recurrent Neural Networks (RNN) with IMDB Dataset  

This repository contains the implementation of a sentiment analysis project using Recurrent Neural Networks (RNN) and the IMDB movie review dataset. The project was developed as part of an assignment to explore the concepts of sentiment analysis, recurrent neural networks, and comparative analysis with other neural network architectures.

## Objective  
The primary objective of this project is to classify IMDB movie reviews as **positive** or **negative** using an RNN model and to compare its performance with a Feedforward Neural Network (FFNN). 

---

## Table of Contents  
1. [Introduction](#introduction)  
2. [Dataset](#dataset)  
3. [Model Implementation](#model-implementation)  
4. [Training and Evaluation](#training-and-evaluation)  
5. [Hyperparameter Tuning](#hyperparameter-tuning)  
6. [Comparative Analysis](#comparative-analysis)  
7. [Results and Insights](#results-and-insights)  


---

## Introduction  
### Sentiment Analysis  
Sentiment analysis involves determining the sentiment (positive or negative) expressed in a piece of text. It has applications in various fields, including customer feedback analysis, social media monitoring, and recommendation systems.  

### Recurrent Neural Networks (RNNs)  
RNNs are a type of neural network designed for sequential data. They utilize hidden states to capture temporal dependencies between data points. However, they can face challenges like vanishing and exploding gradients, making advanced variants such as **LSTM (Long Short-Term Memory)** or **GRU (Gated Recurrent Unit)** more effective.  

---

## Dataset  
The [IMDB movie review dataset](https://www.tensorflow.org/datasets/catalog/imdb_reviews) was used for this project. The dataset contains 25,000 training and 25,000 test reviews labeled as positive or negative.  

### Preprocessing Steps  
1. **Tokenization**: Convert text to numerical tokens.  
2. **Padding**: Uniform input length across sequences for compatibility with the model.  

---

## Model Implementation  
### RNN Model Architecture  
- **Input Layer**: For textual input.  
- **Embedding Layer**: To convert tokens to dense vectors.  
- **LSTM/GRU Layer**: To capture temporal dependencies in data.  
- **Fully Connected Layer**: For feature extraction.  
- **Output Layer**: Sigmoid activation for binary classification.  

### FFNN Model Architecture  
An alternative Feedforward Neural Network (FFNN) was also implemented for comparison, consisting of:  
- Flattened input sequences.  
- Fully connected dense layers.  
- Sigmoid output layer for classification.  

---

## Training and Evaluation  
### Training Process  
- Dataset split into **training** and **validation** sets.  
- Early stopping applied to prevent overfitting.  

### Evaluation Metrics  
- **Accuracy** and **Loss** on training and validation datasets.  
- Visualizations of training and validation metrics over epochs.  

---

## Hyperparameter Tuning  
Various hyperparameters were tested, including:  
- Number of LSTM/GRU units.  
- Dropout rate.  
- Learning rate.  
- Number of layers.  

Each adjustment was documented, and its impact on model performance was analyzed.  

---

## Comparative Analysis  
### Key Findings  
- **RNN (LSTM/GRU)**: Demonstrated strong ability to capture temporal dependencies, suitable for sequential data.  
- **FFNN**: Performed adequately but struggled with sequential nuances, achieving slightly lower accuracy compared to RNN.  

---

## Results and Insights  
- **RNN Performance**: High accuracy, better suited for text data with temporal dependencies.  
- **FFNN Performance**: Simpler architecture with faster training but less effective for sequential text analysis.  

---

