# Drishti Edge 👁️

AI-Powered Diabetic Retinopathy Screening for Low-Connectivity Healthcare

## 📌 About the Project

Drishti Edge is an AI-based diabetic retinopathy screening system that analyzes retinal fundus images and predicts the severity of diabetic retinopathy.

The system is designed with edge-based local inference to reduce dependency on continuous internet connectivity, making it suitable for low-connectivity and resource-limited healthcare environments.

## 🎯 Problem

Diabetic retinopathy can lead to vision loss if it is not detected early. In rural and resource-limited areas, screening can be challenging due to limited specialist availability and unreliable internet connectivity.

Drishti Edge aims to provide an AI-assisted preliminary screening solution using retinal fundus images.

## 💡 Solution

Our system uses a deep learning model to analyze a fundus image and classify it into five diabetic retinopathy severity categories.

The trained model can perform inference locally, supporting screening even when continuous internet connectivity is unavailable.

## 🧠 AI Model

We use:

- Deep Learning
- Convolutional Neural Network (CNN)
- MobileNetV3-Large
- Transfer Learning
- Softmax Classification

### Classification Classes

1. Healthy
2. Mild DR
3. Moderate DR
4. Proliferate DR
5. Severe DR

## ⚙️ System Workflow

Fundus Image
→ Image Preprocessing
→ MobileNetV3-Large
→ Feature Extraction
→ 5-Class Classification
→ Softmax
→ Predicted Condition + Confidence Score

## 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- FastAPI
- HTML
- CSS
- JavaScript
- MobileNetV3-Large

## 🌐 System Architecture

The project consists of:

### Frontend
A web-based interface where the user uploads a retinal fundus image and views the screening result.

### Backend
A FastAPI-based backend that preprocesses the image and performs inference using the trained MobileNetV3-Large model.

### Edge Inference
The trained model is stored locally and can perform prediction without requiring continuous cloud connectivity.

## 🚀 Key Features

- AI-based retinal image screening
- Five-class diabetic retinopathy classification
- Confidence score for predictions
- Lightweight MobileNetV3-Large architecture
- Local inference for low-connectivity environments
- Simple and user-friendly interface

## ⚠️ Limitations

This project is an AI-assisted screening prototype and is not intended to replace professional medical diagnosis.

Further validation with larger and diverse clinical datasets and evaluation by healthcare professionals would be required before real-world clinical deployment.

## 🔮 Future Scope

- Improved model accuracy through larger and more diverse datasets
- Automatic retinal image quality assessment
- Further edge/mobile optimization
- Explainable AI features
- Extensive clinical validation
- Deployment on dedicated edge/mobile devices

## 👥 Project

**Drishti Edge**

An AI-assisted approach towards accessible diabetic retinopathy screening in low-connectivity healthcare environments.
