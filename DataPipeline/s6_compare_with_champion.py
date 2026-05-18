from clearml import Task, Model
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PROJECT_NAME = "NutritionAnalyser"


def get_model_tags(model):
    tags = model.tags or []
    return list(tags)


def set_model_tags(model, add_tags=None, remove_tags=None):
    current_tags = get_model_tags(model)

    if remove_tags:
        current_tags = [
            tag for tag in current_tags
            if tag not in remove_tags
        ]

    if add_tags:
        current_tags.extend(add_tags)

    current_tags = list(set(current_tags))

    model.tags = current_tags

    logger.info(
        f"Updated tags for model {model.id}: {current_tags}"
    )


def get_current_champion_model():
    champion_models = Model.query_models(
        project_name=PROJECT_NAME,
        tags=["champion"]
    )

    if not champion_models:
        return None

    # Use the latest champion model if more than one exists
    champion_models = sorted(
        champion_models,
        key=lambda model: model.created,
        reverse=True
    )

    return champion_models[0]


def get_model_mse_from_metadata(model):
    metadata = model.get_all_metadata() or {}

    if "mse" not in metadata:
        raise KeyError(
            f"Model {model.id} does not have 'mse' metadata. "
            f"Available metadata keys: {list(metadata.keys())}"
        )

    return float(metadata["mse"])


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

    # =========================
    # Load challenger evaluation
    # =========================

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

    challenger_model_id = (
        challenger_result["output_model_id"]
    )

    challenger_model = Model(
        model_id=challenger_model_id
    )

    logger.info(
        f"challenger_model_name: {challenger_model_name}"
    )

    logger.info(
        f"challenger_model_id: {challenger_model_id}"
    )

    logger.info(
        f"challenger_mse: {challenger_mse}"
    )

    # =========================
    # Find current champion
    # =========================

    champion_model = get_current_champion_model()

    if champion_model is None:

        logger.info(
            "No current champion found. "
            "Challenger will be promoted."
        )

        promote = True
        champion_model_id = None
        champion_mse = None

    else:

        champion_model_id = champion_model.id
        champion_mse = get_model_mse_from_metadata(
            champion_model
        )

        logger.info(
            f"Current champion model id: {champion_model_id}"
        )

        logger.info(
            f"Current champion mse: {champion_mse}"
        )

        promote = challenger_mse < champion_mse

    # =========================
    # Promotion decision
    # =========================

    if promote:

        logger.info(
            "Challenger promoted to champion."
        )

        if champion_model is not None:
            set_model_tags(
                champion_model,
                remove_tags=["champion"],
                add_tags=["archived_champion"]
            )

        set_model_tags(
            challenger_model,
            remove_tags=[
                "rejected_challenger",
                "archived_champion"
            ],
            add_tags=[
                "champion",
                "challenger"
            ]
        )

        decision = "promoted"

    else:

        logger.info(
            "Current champion remains champion."
        )

        set_model_tags(
            challenger_model,
            remove_tags=["champion"],
            add_tags=["rejected_challenger", "challenger"]
        )

        decision = "rejected"

    # =========================
    # Upload comparison result
    # =========================

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