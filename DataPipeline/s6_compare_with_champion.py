from clearml import Task, Model
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PROJECT_NAME = "NutritionAnalyser"


def get_current_champion():

    champion_models = Model.query_models(
        project_name=PROJECT_NAME,
        tags=["champion"]
    )

    if not champion_models:
        return None

    champion_models = sorted(
        champion_models,
        key=lambda model: model.created,
        reverse=True
    )

    return champion_models[0]


def update_model_tags(
    model_id,
    add_tags=None,
    remove_tags=None
):

    model = Model(model_id=model_id)

    current_tags = model.tags or []

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
        f"Updated tags for model {model_id}: {current_tags}"
    )


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

        logger.info(
            "Missing task IDs. Template task only."
        )

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

    champion_model = get_current_champion()

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

        champion_metadata = (
            champion_model.get_all_metadata()
        )

        champion_mse = float(
            champion_metadata["mse"]
        )

        logger.info(
            f"Current champion model id: "
            f"{champion_model_id}"
        )

        logger.info(
            f"Current champion mse: "
            f"{champion_mse}"
        )

        promote = challenger_mse < champion_mse

    # =========================
    # Promote challenger
    # =========================

    if promote:

        logger.info(
            "Challenger promoted to champion."
        )

        # Archive old champion
        if champion_model_id is not None:

            update_model_tags(
                model_id=champion_model_id,
                remove_tags=["champion"],
                add_tags=["archived_champion"]
            )

        # Promote challenger
        update_model_tags(
            model_id=challenger_model_id,
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

    # =========================
    # Keep current champion
    # =========================

    else:

        logger.info(
            "Current champion remains champion."
        )

        update_model_tags(
            model_id=challenger_model_id,
            remove_tags=["champion"],
            add_tags=[
                "rejected_challenger",
                "challenger"
            ]
        )

        decision = "rejected"

    # =========================
    # Save comparison result
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

    logger.info(
        "Process completed successfully"
    )


if __name__ == "__main__":
    main()