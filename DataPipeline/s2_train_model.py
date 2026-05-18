from clearml import Task
import pandas as pd
import logging
import math
import os

from data_module import create_dataloader
from model_module import build_model, get_training_components, train, save_model, eval

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
# Initialize the task
    task = Task.init(
        project_name='NutritionAnalyser',
        task_name='s2_train_model',
    )

    args = {
        "preprocess_task_id": "",
        "model_name": "resnet18", # will be replaced by parameter in pipeline
        'num_epochs': 5, # will be replaced by parameter in pipeline
        'batch_size': 32, # will be replaced by parameter in pipeline
        'learning_rate': 1e-3, # will be replaced by parameter in pipeline
        'weight_decay': 1e-5 # will be replaced by parameter in pipeline
    }
    task.connect(args)
    logger.info(f"Connected parameters: {args}")

    preprocess_task_id = task.get_parameter("General/preprocess_task_id")

    # Check if preprocess_task_id is provided, if not, exit the function (this allows us to create a base task template without running the full training)
    if not preprocess_task_id:
        logger.info("No preprocess_task_id provided. This run is only for creating a base task template.")
        return
    
    model_name = task.get_parameter("General/model_name")
    num_epochs = int(task.get_parameter("General/num_epochs"))
    batch_size = int(task.get_parameter("General/batch_size"))
    learning_rate = float(task.get_parameter("General/learning_rate"))
    weight_decay = float(task.get_parameter("General/weight_decay"))

    logger.info(f"preprocess_task_id: {preprocess_task_id}")
    logger.info(f"model_name: {model_name}")
    logger.info(f"num_epochs: {num_epochs}")
    logger.info(f"batch_size: {batch_size}")
    logger.info(f"learning_rate: {learning_rate}")
    logger.info(f"weight_decay: {weight_decay}")

    s1_task = Task.get_task(task_id=preprocess_task_id)

    # load train_df and test_df from input artifacts
    train_df_path = s1_task.artifacts["train_df"].get_local_copy()
    test_df_path = s1_task.artifacts["test_df"].get_local_copy()

    train_df = pd.read_csv(train_df_path)
    test_df = pd.read_csv(test_df_path)

    logger.info("Finished loading preprocessed data.")
    logger.info(f"train_df shape: {train_df.shape}")
    logger.info(f"test_df shape: {test_df.shape}")

    # create dataloader
    train_loader, test_loader = create_dataloader(train_df, test_df, batch_size=batch_size)
    logger.info("Created dataloaders for training and testing.")

    # initialize model and training components
    model, device = build_model(model_name)
    criterion, optimizer = get_training_components(model, learning_rate, weight_decay)

    # train the model
    train(model, train_loader, criterion, optimizer, device, num_epochs=num_epochs)
    logger.info("Finished training the model.")

    evaluation_result = eval(test_loader, model, criterion, device)
    logger.info(f"Evaluation result: {evaluation_result}")

    if "mse" not in evaluation_result:
        raise KeyError(
            f"'mse' was not found in evaluation_result. "
            f"Available keys: {list(evaluation_result.keys())}"
        )

    mse = float(evaluation_result["mse"])

    if math.isnan(mse) or math.isinf(mse):
        raise ValueError(f"Invalid MSE value: {mse}")

    # Report HPO objective metric clearly.
    # ClearML HPO reads title='metrics', series='mse'.
    # Report at both 0 and num_epochs to avoid missing iteration issues.
    task.get_logger().report_scalar(
        title="metrics",
        series="mse",
        value=mse,
        iteration=0
    )

    task.get_logger().report_scalar(
        title="metrics",
        series="mse",
        value=mse,
        iteration=num_epochs
    )

    task.set_parameter("General/mse", mse)

    logger.info(f"HPO objective MSE: {mse}")

    model_dir = "artifacts/model"
    os.makedirs(model_dir, exist_ok=True)

    model_path = f"{model_dir}/{model_name}.pth"

    save_model(model, model_path)

    logger.info(f"Saved the trained model to: {model_path}")

    task.upload_artifact(
        name="model",
        artifact_object=model_path,
        wait_on_upload=True
    )

    task.upload_artifact(
        name="model_metadata",
        artifact_object={
            "model_name": model_name,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "mse": mse,
            "model_path": model_path
        },
        wait_on_upload=True
    )

    task.flush(wait_for_uploads=True)

    logger.info(f"s2_train_model completed for model: {model_name}")


if __name__ == "__main__":
    main()