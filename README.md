# Federated Learning for Chest X-Ray Image Classification using CNN

This project demonstrates a basic Federated Learning (FL) framework for chest X-ray image classification using Convolutional Neural Networks (CNNs). 

Two client models are trained independently on separate data splits, and their learned weights are aggregated using the Federated Averaging (FedAvg) algorithm to create a global model.

The implementation simulates decentralized learning in a Google Colab environment.

---

## Features
- Image-based medical classification (Chest X-ray dataset)
- Data preprocessing using ImageDataGenerator
- Two independent client CNN models
- Local training on separate data splits
- Federated Averaging (FedAvg) weight aggregation
- Global model creation from aggregated weights
- Accuracy evaluation of global model

---

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Google Colab
- ImageDataGenerator
- Zipfile (dataset extraction)

---

## Methodology

### 1. Dataset Handling
- Upload CHEST DATA.zip
- Extract dataset in Colab environment
- Load images from directory structure
- Resize images to 128x128
- Normalize pixel values (rescale = 1/255)

### 2. Data Distribution (Simulated Clients)
- Dataset split into:
  - Client 1 → Training subset
  - Client 2 → Validation subset
- Each client trains independently

### 3. CNN Architecture
- Conv2D (16 filters, 3x3)
- MaxPooling
- Conv2D (32 filters, 3x3)
- MaxPooling
- Flatten
- Dense (64 units)
- Output layer (Softmax)

Loss Function:
- Categorical Crossentropy

Optimizer:
- Adam

### 4. Local Training
- Each client model trains for 3 epochs
- Weights are updated independently

### 5. Federated Averaging (FedAvg)
- Extract weights from both client models
- Compute element-wise average of weights
- Initialize a new global model
- Assign averaged weights to global model

### 6. Evaluation
- Evaluate global model performance
- Report classification accuracy

---

## Federated Learning Pipeline

- Dataset Upload
- Client 1 Training
- Client 2 Training
- Weight Extraction
- Federated Averaging
- Global Model Creation
- Global Model Evaluation

---

## Results
- Successfully simulated federated learning setup
- Observed weight updates after local training
- Aggregated weights using FedAvg
- Global model evaluated on client data
- Demonstrated decentralized learning workflow

---

## Project Structure

federated-chest-xray-classification/
│
├── CHEST DATA.zip
├── federated_training.py
├── dataset/
└── README.md

---

## How to Run

1. Open Google Colab
2. Upload CHEST DATA.zip when prompted
3. Run the script
4. Observe:
   - Initial weights
   - Post-training weights
   - Global aggregated weights
   - Final accuracy

---

## Key Concept Demonstrated

This project illustrates the core principle of Federated Learning:

- Data remains distributed
- Models train locally
- Only weights are shared
- Global model is formed through aggregation

---

## Future Improvements

- Add multiple federated rounds
- Introduce non-IID data distribution
- Use weighted FedAvg
- Implement secure aggregation
- Add validation on unseen test set
- Extend to real hospital-level simulation

---

## Educational Purpose

This project serves as a practical demonstration of Federated Learning fundamentals using CNNs in a medical imaging context.
