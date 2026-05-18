from clearml import Task, Model
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

    preprocess_task_id = task.get_parameter(
        "General/preprocess_task_id"
    )

    best_model_task_id = task.get_parameter(
        "General/best_model_task_id"
    )

    batch_size = int(
        task.get_parameter("General/batch_size")
    )

    if not preprocess_task_id or not best_model_task_id:

        logger.info(
            "Missing preprocess_task_id or best_model_task_id. "
            "This run is only for creating a base task template."
        )

        return

    # =========================
    # Load dataset
    # =========================

    s1_task = Task.get_task(
        task_id=preprocess_task_id
    )

    train_df_path = (
        s1_task
        .artifacts["train_df"]
        .get_local_copy()
    )

    test_df_path = (
        s1_task
        .artifacts["test_df"]
        .get_local_copy()
    )

    train_df = pd.read_csv(train_df_path)
    test_df = pd.read_csv(test_df_path)

    logger.info(
        "Loaded train_df and test_df from s1 artifacts."
    )

    _, test_loader = create_dataloader(
        train_df,
        test_df,
        batch_size=batch_size
    )

    logger.info(
        "Created test dataloader."
    )

    # =========================
    # Load s4 metadata
    # =========================

    best_model_task = Task.get_task(
        task_id=best_model_task_id
    )

    model_metadata = (
        best_model_task
        .artifacts["model_metadata"]
        .get()
    )

    model_name = model_metadata["model_name"]

    output_model_id = model_metadata["output_model_id"]

    logger.info(f"Model name: {model_name}")
    logger.info(f"OutputModel ID: {output_model_id}")

    # =========================
    # Load model from registry
    # =========================

    registered_model = Model(
        model_id=output_model_id
    )

    model_path = registered_model.get_local_copy()

    logger.info(
        f"Loaded model weights from registry: {model_path}"
    )

    # =========================
    # Build model
    # =========================

    model, device = build_model(
        model_name
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    criterion = torch.nn.MSELoss()

    # =========================
    # Evaluate model
    # =========================

    evaluation_result = eval(
        test_loader,
        model,
        criterion,
        device
    )

    logger.info(
        f"Evaluation result: {evaluation_result}"
    )

    evaluation_artifact = {
        "model_name": model_name,
        "best_model_task_id": best_model_task_id,
        "output_model_id": output_model_id
    }

    evaluation_artifact.update({
        metric_name: float(metric_value)
        for metric_name, metric_value
        in evaluation_result.items()
    })

    # =========================
    # Log metrics
    # =========================

    for metric_name, metric_value in evaluation_artifact.items():

        if isinstance(metric_value, (int, float)):

            task.get_logger().report_scalar(
                title="final_evaluation",
                series=metric_name,
                value=float(metric_value),
                iteration=0
            )

    # =========================
    # Upload evaluation result
    # =========================

    task.upload_artifact(
        name="evaluation_result",
        artifact_object=evaluation_artifact,
        wait_on_upload=True
    )

    task.flush(wait_for_uploads=True)

    logger.info("Final evaluation metrics:")

    for metric_name, metric_value in evaluation_artifact.items():
        logger.info(f"{metric_name}: {metric_value}")

    logger.info(
        "s5_evaluate_model completed."
    )


if __name__ == "__main__":
    main()