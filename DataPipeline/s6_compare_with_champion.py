from clearml import Task
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_current_champion():
    champions = Task.get_tasks(
        project_name="NutritionAnalyser",
        task_name="s4_train_best_model",
        tags=["__$and", "champion"]
    )

    if not champions:
        return None

    # เลือก champion ล่าสุด
    champions = sorted(champions, key=lambda t: t.data.completed or t.data.started, reverse=True)
    return champions[0]


def main():
    task = Task.init(
        project_name="NutritionAnalyser",
        task_name="s6_compare_with_champion"
    )

    args = {
        "evaluation_task_id": "",
        "best_model_task_id": ""
    }
    task.connect(args)

    evaluation_task_id = task.get_parameter("General/evaluation_task_id")
    best_model_task_id = task.get_parameter("General/best_model_task_id")

    if not evaluation_task_id or not best_model_task_id:
        logger.info("Missing task IDs. Template task only.")
        return

    # challenger evaluation
    evaluation_task = Task.get_task(task_id=evaluation_task_id)
    challenger_result = evaluation_task.artifacts["evaluation_result"].get()

    challenger_mse = float(challenger_result["mse"])
    challenger_model_name = challenger_result["model_name"]

    challenger_task = Task.get_task(task_id=best_model_task_id)

    # current champion
    champion_task = get_current_champion()

    if champion_task is None:
        promote = True
        champion_mse = None
        champion_task_id = None 
        logger.info("No current champion found. Challenger will be promoted.")
    else:
        champion_metadata = champion_task.artifacts["model_metadata"].get()
        champion_mse = float(champion_metadata["mse"])
        champion_task_id = champion_task.id

        promote = challenger_mse < champion_mse

    if promote:
        challenger_task.add_tags(["champion"])

        if champion_task is not None:
            champion_task.add_tags(["archived_champion"])

        decision = "promoted"
        logger.info("Challenger promoted to champion.")
    else:
        challenger_task.add_tags(["rejected_challenger"])
        decision = "rejected"
        logger.info("Current champion remains champion.")

    comparison_result = {
        "decision": decision,
        "challenger_task_id": best_model_task_id,
        "challenger_model_name": challenger_model_name,
        "challenger_mse": challenger_mse,
        "previous_champion_task_id": champion_task_id,
        "previous_champion_mse": champion_mse
    }

    task.upload_artifact(
        name="comparison_result",
        artifact_object=comparison_result,
        wait_on_upload=True
    )

    logger.info(comparison_result)


if __name__ == "__main__":
    main()