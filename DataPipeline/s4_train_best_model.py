from clearml import Task
import pandas as pd
import logging

from data_module import create_dataloader
from model_module import build_model, get_training_components, train, save_model, eval

import torch
from clearml.automation import PipelineController

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    task = Task.init(
        project_name="NutritionAnalyser",
        task_name="s4_train_best_model"
    )

    args = {
        "preprocess_task_id": "",
        "hpo_task_id": "",
        "num_epochs": 100
    }

    task.connect(args)
    logger.info(f"Connected parameters: {args}")

    preprocess_task_id = task.get_parameter("General/preprocess_task_id")
    hpo_task_id = task.get_parameter("General/hpo_task_id")

    if not preprocess_task_id or not hpo_task_id:
        logger.info(
            "Missing preprocess_task_id or hpo_task_id. "
            "This run is only for creating a base task template."
        )
        return

    s3_hpo_task = Task.get_task(task_id=hpo_task_id)
    best_hpo_result = s3_hpo_task.artifacts["best_hpo_result"].get()

    best_train_task_id = best_hpo_result["best_train_task_id"]
    best_train_task = Task.get_task(task_id=best_train_task_id)

    best_model_name = best_train_task.get_parameter("General/model_name")
    best_batch_size = int(best_train_task.get_parameter("General/batch_size"))
    best_learning_rate = float(best_train_task.get_parameter("General/learning_rate"))
    best_weight_decay = float(best_train_task.get_parameter("General/weight_decay"))

    logger.info(f"Best model name: {best_model_name}")
    logger.info(f"Best batch size: {best_batch_size}")
    logger.info(f"Best learning rate: {best_learning_rate}")
    logger.info(f"Best weight decay: {best_weight_decay}")

    s1_task = Task.get_task(task_id=preprocess_task_id)

    train_df_path = s1_task.artifacts["train_df"].get_local_copy()
    test_df_path = s1_task.artifacts["test_df"].get_local_copy()

    train_df = pd.read_csv(train_df_path)
    test_df = pd.read_csv(test_df_path)

    logger.info("Loaded train_df and test_df from s1 artifacts.")

    train_loader, test_loader = create_dataloader(
        train_df,
        test_df,
        batch_size=best_batch_size
    )

    model, device = build_model(best_model_name)

    criterion, optimizer = get_training_components(
        model,
        best_learning_rate,
        best_weight_decay
    )

    train(model, train_loader, criterion, optimizer, device, num_epochs=args["num_epochs"])
    logger.info("Finished training best model.")

    evaluation_result = eval(
        test_loader,
        model,
        torch.nn.MSELoss(),
        device
    )

    mse = float(evaluation_result["mse"])

    task.get_logger().report_scalar(
        title="metrics",
        series="mse",
        value=mse,
        iteration=args["num_epochs"]
    )

    model_path = f"artifacts/model/best_{best_model_name}.pth"
    save_model(model, model_path)

    task.add_tags(["challenger"])

    task.upload_artifact(
        name="model",
        artifact_object=model_path,
        wait_on_upload=True
    )

    pipeline_model = PipelineController.upload_model(
        model_name=f"challenger_{best_model_name}",
        model_local_path=model_path
    )

    PipelineController.upload_artifact(
        name="challenger_metadata",
        artifact_object={
            "challenger_task_id": task.id,
            "model_name": best_model_name,
            "mse": mse,
            "model_path": model_path
        },
        wait_on_upload=True
    )

    task.upload_artifact(
        name="model_metadata",
        artifact_object={
            "source": "hpo_best_model",
            "best_hpo_train_task_id": best_train_task_id,
            "model_name": best_model_name,
            "num_epochs": args["num_epochs"],
            "batch_size": best_batch_size,
            "learning_rate": best_learning_rate,
            "weight_decay": best_weight_decay,
            "mse": mse,
            "model_path": model_path
        },
        wait_on_upload=True
    )

    logger.info(f"s4_train_best_model completed. Final MSE: {mse}")


if __name__ == "__main__":
    main()