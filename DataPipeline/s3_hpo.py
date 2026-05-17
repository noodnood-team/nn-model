from clearml import Task
from clearml.automation import HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna
from clearml.automation.parameters import DiscreteParameterRange
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    task = Task.init(
        project_name="NutritionAnalyser",
        task_name="s3_hpo"
    )

    args = {
        "preprocess_task_id": "",
        "base_train_task_id": "",
        "max_number_of_experiments": 6,
        "max_concurrent_tasks": 1,
        "execution_queue": "default"
    }

    task.connect(args)
    logger.info(f"Connected parameters: {args}")

    preprocess_task_id = task.get_parameter("General/preprocess_task_id")
    base_train_task_id = task.get_parameter("General/base_train_task_id")

    if not preprocess_task_id or not base_train_task_id:
        logger.info(
            "Missing preprocess_task_id or base_train_task_id. "
            "This run is only for creating a base task template."
        )
        return

    logger.info(f"Using preprocess task ID: {preprocess_task_id}")
    logger.info(f"Using base train task ID: {base_train_task_id}")

    optimizer = HyperParameterOptimizer(
        base_task_id=base_train_task_id,
        hyper_parameters=[
            DiscreteParameterRange(
                "General/preprocess_task_id",
                values=[preprocess_task_id]
            ),
            DiscreteParameterRange(
                "General/model_name",
                values=["resnet18", "resnet34", "mobilenet_v3_small"]
            ),
            DiscreteParameterRange(
                "General/num_epochs",
                values=[10]
            ),
            DiscreteParameterRange(
                "General/batch_size",
                values=[16, 32]
            ),
            DiscreteParameterRange(
                "General/learning_rate",
                values=[1e-3, 1e-4]
            ),
            DiscreteParameterRange(
                "General/weight_decay",
                values=[1e-5, 1e-4]
            ),
        ],
        objective_metric_title="metrics",
        objective_metric_series="mse",
        objective_metric_sign="min",
        optimizer_class=OptimizerOptuna,
        execution_queue=args["execution_queue"],
        max_number_of_concurrent_tasks=args["max_concurrent_tasks"],
        total_max_jobs=args["max_number_of_experiments"],
        max_iteration_per_job=1,
    )

    optimizer.set_report_period(1)

    logger.info("Starting HPO...")
    optimizer.start()
    optimizer.wait()
    optimizer.stop()

    top_experiments = optimizer.get_top_experiments(top_k=1)

    if not top_experiments:
        raise RuntimeError("No HPO experiments were completed.")

    best_task = top_experiments[0]

    logger.info(f"Best HPO task ID: {best_task.id}")
    logger.info(f"Best HPO task name: {best_task.name}")

    task.set_parameter("General/best_train_task_id", best_task.id)

    task.upload_artifact(
        name="best_hpo_result",
        artifact_object={
            "best_train_task_id": best_task.id,
            "best_train_task_name": best_task.name,
            "objective_metric_title": "metrics",
            "objective_metric_series": "mse",
            "objective_metric_sign": "min"
        },
        wait_on_upload=True
    )

    logger.info("s3_hpo completed.")


if __name__ == "__main__":
    main()