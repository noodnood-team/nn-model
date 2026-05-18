from clearml import Task
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PROJECT_NAME = "NutritionAnalyser"


def get_current_champion():
    champions = Task.get_tasks(
        project_name=PROJECT_NAME,
        task_name="s4_train_best_model",
        tags=["__$and", "champion"]
    )

    if not champions:
        return None

    champions = sorted(
        champions,
        key=lambda t: t.data.completed or t.data.started,
        reverse=True
    )

    return champions[0]


def update_task_tags(task_id, add_tags=None, remove_tags=None):
    task = Task.get_task(task_id=task_id)

    current_tags = list(task.get_tags() or [])

    if remove_tags:
        current_tags = [
            tag for tag in current_tags
            if tag not in remove_tags
        ]

    if add_tags:
        current_tags.extend(add_tags)

    current_tags = list(set(current_tags))

    task.set_tags(current_tags)

    logger.info(f"Updated tags for task {task_id}: {current_tags}")

def main():
    task = Task.init(
        project_name=PROJECT_NAME,
        task_name="s6_compare_with_champion"
    )

    args = {
        "evaluation_task_id": "",
        "best_model_task_id": ""
    }

    task.connect(args)

    evaluation_task_id = task.get_parameter(
        "General/evaluation_task_id"
    )

    best_model_task_id = task.get_parameter(
        "General/best_model_task_id"
    )

    if not evaluation_task_id or not best_model_task_id:
        logger.info("Missing task IDs. Template task only.")
        return

    logger.info(
        f"evaluation_task_id: {evaluation_task_id}"
    )

    logger.info(
        f"best_model_task_id: {best_model_task_id}"
    )

    # challenger evaluation
    evaluation_task = Task.get_task(
        task_id=evaluation_task_id
    )

    challenger_result = (
        evaluation_task
        .artifacts["evaluation_result"]
        .get()
    )

    challenger_mse = float(
        challenger_result["mse"]
    )

    challenger_model_name = (
        challenger_result["model_name"]
    )

    challenger_task = Task.get_task(
        task_id=best_model_task_id
    )

    logger.info(
        f"challenger_model_name: {challenger_model_name}"
    )

    logger.info(
        f"challenger_mse: {challenger_mse}"
    )

    # current champion
    champion_task = get_current_champion()

    if champion_task is None:

        logger.info(
            "No current champion found. "
            "Challenger will be promoted."
        )

        promote = True
        champion_mse = None
        champion_task_id = None

    else:
        champion_metadata = (
            champion_task
            .artifacts["model_metadata"]
            .get()
        )

        champion_mse = float(
            champion_metadata["mse"]
        )

        champion_task_id = champion_task.id

        logger.info(
            f"Current champion task: {champion_task_id}"
        )

        logger.info(
            f"Current champion mse: {champion_mse}"
        )

        promote = challenger_mse < champion_mse

    # challenger wins
    if promote:

        logger.info(
            "Challenger promoted to champion."
        )

        # archive old champion
        if champion_task is not None:

            update_task_tags(
                task_id=champion_task_id,
                remove_tags=["champion"],
                add_tags=["archived_champion"]
            )

        # promote challenger
        update_task_tags(
            task_id=best_model_task_id,
            remove_tags=[
                "rejected_challenger",
                "archived_champion"
            ],
            add_tags=["champion"]
        )

        decision = "promoted"

    # current champion still better
    else:

        logger.info(
            "Current champion remains champion."
        )

        update_task_tags(
            task_id=best_model_task_id,
            add_tags=["rejected_challenger"]
        )

        decision = "rejected"

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

    logger.info("Process completed successfully")


if __name__ == "__main__":
    main()