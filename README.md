# BrainTumorDetection_MLOPS

## Overview
BrainTumorDetection_MLOPS is a FastAPI-based brain tumor detection system designed with MLOps principles for efficient model deployment, automated data handling, and seamless API integration. The system uses MLflow for model tracking, DVC for data versioning, and DAGsHub as the central server for both MLflow and DVC. It employs a Vision Transformer (ViT) model for accurate tumor detection from medical images.

## Features
- **FastAPI-based API** for real-time predictions.
- **MLflow integration** to track model performance and experiments.
- **DVC (Data Version Control)** for handling datasets efficiently.
- **DAGsHub as a unified platform** for MLflow and DVC.
- **Automated pipeline** for data processing and deployment.

## Tech Stack
- **FastAPI** - Web framework for serving the model.
- **MLflow** - Model tracking and experiment logging.
- **DVC** - Data versioning and management.
- **DAGsHub** - Hosting and integration for MLflow and DVC.

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/BrainTumorDetection_MLOPS.git
cd BrainTumorDetection_MLOPS
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure MLflow & DVC
#### Initialize DVC and Add Remote Storage
```bash
dvc init
dvc remote add origin https://dagshub.com/yourusername/BrainTumorDetection_MLOPS.dvc
```
#### Set Up MLflow Tracking
```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/yourusername/BrainTumorDetection_MLOPS.mlflow
```

### 5. Run the FastAPI Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### **Predict Tumor**
- **Endpoint:** `/docs`
- **Method:** `POST`
- **Payload:**
  ```json
  {
    "image": "base64_encoded_image"
  }
  ```
- **Response:**
  ```json
  {
    "prediction": "No abnormal growth detected. However, consult a doctor for confirmation"
  }
  ```

## Contributing
Feel free to contribute.

## License
This project is licensed under the MIT License.

