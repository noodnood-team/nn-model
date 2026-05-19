🐧 Noodnood | Nutrition Fuel Estimation

This project builds an automated machine learning workflow for food nutrition prediction from images.

The system supports multiple CNN regression models including ResNet18, ResNet34, and MobileNetV3 Small for food nutrition prediction.

- Calories
- Protein
- Carbohydrates
- Fat

The project also includes a Flask API service for model inference.

## Project Overview

The workflow is split into three main ClearML tasks:

1. Data preprocessing
2. Model training
3. Hyperparameter optimization (HPO)
4. Training the best model with optimized hyperparameters
5. Model evaluation
6. Comparison with the champion model

Each step runs as a separate ClearML task and can be connected through a ClearML pipeline.

## Project Structure

```text
nn-project/
│
├── .github/workflows/
│   ├── deploy_model_service.yml   # GitHub Actions workflow for deploying the model service
│   └── run_data_pipeline.yml      # GitHub Actions workflow for running the data pipeline

│
├── DataPipeline/
│   ├── clearml_pipeline.py        # Main ClearML pipeline controller
│   ├── s1_data_preprocessing.py   # Loads, cleans, engineers features, and splits data
│   ├── s2_train_model.py          # Trains the ResNet18 model
│   ├── s3_hpo.py                  # Hyperparameter optimization
│   ├── s4_train_best_model.py     # Trains the best model with optimized hyperparameters
│   ├── s5_evaluate_model.py       # Evaluates the trained model
│   ├── s6_compare_with_champion.py # Compares the new model with the champion model
│   ├── data_module.py             # Data loading, preprocessing, dataset, and dataloader logic
│   ├── model_module.py            # Model, training, saving, and evaluation functions
│   └── requirements.txt           # Dependencies for the data pipeline
│
├── model_service/
│   ├── app.py                     # Flask API for model inference
│   ├── load_model.py              # Utilities to load the champion weights from ClearML
│   └── requirements.txt           # Dependencies for the model service
│
└── README.md
```

## How to Run

### 1. How to Run Data Pipeline

There are two ways to run the pipeline:

- Run each step manually (for first-time setup)
- Run the full pipeline using ClearML (recommended, after initial setup)

#### Step 1: Run Each Step Manually

This step is required for the first-time setup to create base tasks in ClearML.

1. Navigate to the `DataPipeline` directory:
   ```bash
   cd DataPipeline
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt  
   ```

3. Run all base tasks first:
   ```bash
   python s1_data_preprocessing.py
   python s2_train_model.py
   python s3_hpo.py
   python s4_train_best_model.py
   python s5_evaluate_model.py
   python s6_compare_with_champion.py
   ```  

This process ensures that all base tasks are correctly registered in ClearML before running the full pipeline.

#### Step 2: Run the Full Pipeline
After the initial setup, you can run the entire pipeline with a single command:

```bash
python clearml_pipeline.py
```

This will execute all steps in the correct order, with ClearML handling task dependencies and tracking.

##### You need to start 2 workers to run the pipeline. You can start them in separate terminal windows:
worker 1 (for controller tasks):
```bash
   CLEARML_WORKER_ID=hpo-trainer clearml-agent daemon --queue default
```
worker 2 (for hpo tasks):
```bash
   CLEARML_WORKER_ID=hpo-controller clearml-agent daemon --queue hpo_service
```

### 2. How to Run Model Service in local machine
To run the Flask API for model inference:
1. Navigate to the `model_service` directory:
   ```bash
   cd model_service
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the Flask app:
    ```bash
    python app.py
    ```

The API will be available at `http://localhost:8080`.
API Endpoint:
- `POST /predict`: Accepts an image file and returns predicted nutrition values.
- `GET /health`: Returns a simple health check response.
- `GET /model_info`: Returns information about the currently loaded model.
