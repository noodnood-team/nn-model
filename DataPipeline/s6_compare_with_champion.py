from clearml import Task, Model
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_NAME = "NutritionAnalyser"


def get_model_metadata(model, key, default=None):
    value = model.get_metadata(key)
    if value is None:
        return default
    return value


def set_model_metadata(model, key, value):
    ok = model.set_metadata(key, str(value))
    logger.info(f"Set metadata for model {model.id}: {key}={value}, success={ok}")


def update_model_tags(model, add_tags=None, remove_tags=None):
    current_tags = list(model.tags or [])

    if remove_tags:
        current_tags = [
            tag for tag in current_tags
            if tag not in remove_tags
        ]

    if add_tags:
        current_tags.extend(add_tags)

    current_tags = list(set(current_tags))
    model.tags = current_tags

    logger.info(f"Updated tags for model {model.id}: {current_tags}")


def get_current_champion():
    models = Model.query_models(
        project_name=None,
        tags=None,
        only_published=False,
        include_archived=True,
        max_results=100
    )

    champion_candidates = []

    for model in models:
        role = get_model_metadata(model, "role", "")
        tags = list(model.tags or [])

        if role == "champion" or "champion" in tags:
            champion_candidates.append(model)

    if not champion_candidates:
        return None

    champion_candidates = sorted(
        champion_candidates,
        key=lambda m: m.created,
        reverse=True
    )

    return champion_candidates[0]


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

    evaluation_task_id = task.get_parameter("General/evaluation_task_id")
    best_model_task_id = task.get_parameter("General/best_model_task_id")

    if not evaluation_task_id or not best_model_task_id:
        logger.info("Missing task IDs. Template task only.")
        return

    evaluation_task = Task.get_task(task_id=evaluation_task_id)
    challenger_result = evaluation_task.artifacts["evaluation_result"].get()

    challenger_mse = float(challenger_result["mse"])
    challenger_model_name = challenger_result["model_name"]
    challenger_model_id = challenger_result["output_model_id"]

    challenger_model = Model(model_id=challenger_model_id)

    logger.info(f"challenger_model_name: {challenger_model_name}")
    logger.info(f"challenger_model_id: {challenger_model_id}")
    logger.info(f"challenger_mse: {challenger_mse}")

    champion_model = get_current_champion()

    if champion_model is None:
        logger.info("No current champion found. Challenger will be promoted.")
        promote = True
        champion_model_id = None
        champion_mse = None
    else:
        champion_model_id = champion_model.id
        champion_mse = float(get_model_metadata(champion_model, "mse"))

        logger.info(f"Current champion model id: {champion_model_id}")
        logger.info(f"Current champion mse: {champion_mse}")

        promote = challenger_mse < champion_mse

    if promote:
        logger.info("Challenger promoted to champion.")

        if champion_model is not None:
            set_model_metadata(champion_model, "role", "archived_champion")
            update_model_tags(
                champion_model,
                remove_tags=["champion"],
                add_tags=["archived_champion"]
            )

        set_model_metadata(challenger_model, "role", "champion")
        set_model_metadata(challenger_model, "mse", challenger_mse)
        set_model_metadata(challenger_model, "model_name", challenger_model_name)

        update_model_tags(
            challenger_model,
            remove_tags=["rejected_challenger", "archived_champion"],
            add_tags=["champion", "challenger"]
        )

        decision = "promoted"

    else:
        logger.info("Current champion remains champion.")

        set_model_metadata(challenger_model, "role", "rejected_challenger")
        set_model_metadata(challenger_model, "mse", challenger_mse)
        set_model_metadata(challenger_model, "model_name", challenger_model_name)

        update_model_tags(
            challenger_model,
            remove_tags=["champion"],
            add_tags=["rejected_challenger", "challenger"]
        )

        decision = "rejected"

    comparison_result = {
        "decision": decision,
        "challenger_model_id": challenger_model_id,
        "challenger_task_id": best_model_task_id,
        "challenger_model_name": challenger_model_name,
        "challenger_mse": challenger_mse,
        "previous_champion_model_id": champion_model_id,
        "previous_champion_mse": champion_mse
    }

    task.upload_artifact(
        name="comparison_result",
        artifact_object=comparison_result,
        wait_on_upload=True
    )

    task.flush(wait_for_uploads=True)

    logger.info(comparison_result)
    logger.info("Process completed successfully")


if __name__ == "__main__":
    main()