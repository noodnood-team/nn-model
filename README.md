🐧 Noodnood | Nutrition Fuel Estimation

This project builds an automated machine learning workflow for food nutrition prediction from images.

The system uses ClearML to manage the data pipeline, model training, evaluation, and experiment tracking. The trained model is a ResNet18-based regression model that predicts four nutrition values from a food image:

- Calories
- Protein
- Carbohydrates
- Fat

The project also includes a Flask API service for model inference.

## Project Overview

The workflow is split into three main ClearML tasks:

1. Data preprocessing
2. Model training
3. Model evaluation

Each step runs as a separate ClearML task and can be connected through a ClearML pipeline.

## Project Structure

```text
nn-project/
│
├── DataPipeline/
│   ├── clearml_pipeline.py        # Main ClearML pipeline controller
│   ├── s1_data_preprocessing.py   # Loads, cleans, engineers features, and splits data
│   ├── s2_train_model.py          # Trains the ResNet18 model
│   ├── s3_evaluate_model.py       # Evaluates the trained model
│   ├── data_module.py             # Data loading, preprocessing, dataset, and dataloader logic
│   ├── model_module.py            # Model, training, saving, and evaluation functions
│   └── requirements.txt
│
├── model_service/
│   └── app.py                     # Flask API for model inference
│
├── Data_pipeline.ipynb
├── run.ipynb
├── requirement.txt
└── README.md
```

## Project Overview

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

3. Run preprocessing first:
   ```bash
   python s1_data_preprocessing.py
   ```

4. After the run is completed, copy the `preprocess_task_id` from ClearML.

5. Open `s2_train_model.py` and `s3_evaluate_model.py`, then set:
   ```python
   preprocess_task_id = "your_preprocess_task_id"
   ```

6. Run training:
   ```bash
   python s2_train_model.py
   ```

7. After training is completed, copy the `train_task_id` from ClearML.

8. Open `s3_evaluate_model.py`, then set:
   ```python
   train_task_id = "your_train_task_id"
   ```

9. Run evaluation:
   ```bash
   python s3_evaluate_model.py
   ```

This process ensures that all base tasks are correctly registered in ClearML before running the full pipeline.

After the initial setup is complete, remove the hardcoded task IDs from `s2_train_model.py` and `s3_evaluate_model.py` to allow the pipeline to manage task dependencies automatically.

#### Step 2: Run the Full Pipeline
After the initial setup, you can run the entire pipeline with a single command:

```bash
python clearml_pipeline.py
```

This will execute all steps in the correct order, with ClearML handling task dependencies and tracking.

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
