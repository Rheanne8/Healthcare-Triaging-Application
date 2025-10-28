# Healthcare-Triaging-Application

This project is a Streamlit-based web application that predicts disease and symptom urgency using trained machine learning models. It provides an interactive interface where users can select symptoms and feature values, and receive triage outputs such as predicted disease, urgency, and supporting information.  

# Features
Interactive Streamlit UI with sliders and selectors for patient features.  
Encodes symptom presence/absence from a provided dataset.  
Uses trained Random Forest for disease prediction as well as a deep learning model with multi layer perceptron architecture to predict benign/malignant cancer 
Displays urgency classification and interpretable results.  
Models and encoders are pre-trained and loaded automatically from the models/ folder.  

# Models
The application relies on 2 pre-trained models:
## 1. Disease Classifier
A random forest classifier was trained to predict diseases based on user input symptoms
Trained on: https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset

## 2. Multi-Layer Perception (PyTorch-based)
A small feed-forward neural network was trained to predict if breast cancer is malignant or benign based on 30 features
The model was trained on https://www.kaggle.com/datasets/erdemtaha/cancer-data

# Hosted version
To view this Streamlit application for a demonstration of the healthcare triaging application, please visit:  
https://healthcare-triaging-application.streamlit.app/
