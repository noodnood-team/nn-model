# Nutrition Analyser - Model Inference Service

This directory contains the Flask-based Web API service that serves predictions for food nutrition from images. It dynamically retrieves the champion model's weights and metadata directly from **ClearML** upon startup.

---

## Service Architecture

- **`app.py`**: The Flask application containing endpoint routing, request parsing, and image prediction transformations.
- **`load_model.py`**: Utilizes the ClearML API to locate the current "champion" model registered in your ClearML workspace, downloads its weight files, and builds the PyTorch model architecture dynamically.
- **`uploads/`**: A temporary directory where uploaded images are saved before they are processed by the model and deleted.

---

## API Endpoints

### 1. Health Check
* **Endpoint**: `GET /health`
* **Response**:
  ```json
  {
    "status": "ok"
  }
  ```

### 2. Model Information
* **Endpoint**: `GET /model_info`
* **Description**: Returns details about the current active champion model name, its source pipeline task ID, and evaluation performance metrics (MAE, RMSE, MSE).
* **Response**:
  ```json
  {
    "model_name": "resnet18",
    "pipeline_id": "ab12cd34ef...",
    "model_performance": {
      "overall_mae": 0.12,
      "overall_rmse": 0.15,
      "overall_mse": 0.02,
      "mse_log_loss": 0.025,
      "targets": {
        "calories": { "mae": 45.2, "rmse": 60.1 },
        "protein": { "mae": 3.1, "rmse": 4.5 },
        "carbs": { "mae": 8.5, "rmse": 11.2 },
        "fat": { "mae": 2.1, "rmse": 3.4 }
      }
    }
  }
  ```

### 3. Nutrition Prediction
* **Endpoint**: `POST /predict`
* **Request Format**: Multipart Form Data (`multipart/form-data`)
  * **Key**: `image` (File containing the food image)
* **Response**:
  ```json
  {
    "calories": 245.5,
    "protein": 12.4,
    "carbs": 28.1,
    "fat": 6.8
  }
  ```

---

## How to Run Locally

### 1. Install Dependencies
Ensure you have PyTorch and Flask installed:
```bash
pip install -r requirements.txt
```

### 2. Configure ClearML
Make sure your ClearML credentials are set up. Upon startup, the service queries ClearML to find and download the champion model.

### 3. Run the App
Start the Flask application:
```bash
python app.py
```
The server will start on port `8080`. You can verify it is active by visiting `http://localhost:8080/health`.
