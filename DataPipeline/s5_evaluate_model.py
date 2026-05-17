from clearml import Task
import pandas as pd
import logging
import torch

from data_module import create_dataloader
from model_module import build_model, eval


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    task = Task.init(
        project_name="NutritionAnalyser",
        task_name="s5_evaluate_model"
    )

    args = {
        "preprocess_task_id": "",
        "best_model_task_id": "",
        "batch_size": 32
    }

    task.connect(args)
    logger.info(f"Connected parameters: {args}")

    preprocess_task_id = task.get_parameter("General/preprocess_task_id")
    best_model_task_id = task.get_parameter("General/best_model_task_id")

    if not preprocess_task_id or not best_model_task_id:
        logger.info(
            "Missing preprocess_task_id or best_model_task_id. "
            "This run is only for creating a base task template."
        )
        return

    # Load test data from s1
    s1_task = Task.get_task(task_id=preprocess_task_id)

    train_df_path = s1_task.artifacts["train_df"].get_local_copy()
    test_df_path = s1_task.artifacts["test_df"].get_local_copy()

    train_df = pd.read_csv(train_df_path)
    test_df = pd.read_csv(test_df_path)

    logger.info("Loaded train_df and test_df from s1 artifacts.")

    _, test_loader = create_dataloader(
        train_df,
        test_df,
        batch_size=args["batch_size"]
    )

    logger.info("Created test dataloader.")

    # Load best model from s4
    best_model_task = Task.get_task(task_id=best_model_task_id)

    model_metadata = best_model_task.artifacts["model_metadata"].get()
    model_name = model_metadata["model_name"]

    model_path = best_model_task.artifacts["model"].get_local_copy()

    logger.info(f"Loaded best model artifact from: {model_path}")
    logger.info(f"Evaluating model: {model_name}")

    model, device = build_model(model_name)

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    model.to(device)
    model.eval()

    criterion = torch.nn.MSELoss()

    evaluation_result = eval(
        test_loader,
        model,
        criterion,
        device
    )

    evaluation_artifact = {
        "model_name": model_name,
        "best_model_task_id": best_model_task_id
    }

    evaluation_artifact.update({
        metric_name: float(metric_value)
        for metric_name, metric_value in evaluation_result.items()
    })

    # Log metrics to ClearML scalar charts
    for metric_name, metric_value in evaluation_artifact.items():
        if isinstance(metric_value, (int, float)):
            task.get_logger().report_scalar(
                title="final_evaluation",
                series=metric_name,
                value=float(metric_value),
                iteration=0
            )

    task.upload_artifact(
        name="evaluation_result",
        artifact_object=evaluation_artifact,
        wait_on_upload=True
    )

    logger.info("Final evaluation metrics:")
    for metric_name, metric_value in evaluation_artifact.items():
        logger.info(f"{metric_name}: {metric_value}")

    logger.info("s5_evaluate_model completed.")


if __name__ == "__main__":
    main()