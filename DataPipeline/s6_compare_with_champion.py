from clearml import Task, Model
import logging
import sys


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_NAME = "NutritionAnalyser" # Consider making this dynamic


def get_model_metadata(model, key, default=None):
    """Safely retrieves metadata from a ClearML model."""
    value = model.get_metadata(key)
    if value is None:
        return default
    return value


def set_model_metadata(model, key, value):
    """Sets metadata for a ClearML model and logs the action."""
    str_value = str(value)
    ok = model.set_metadata(key, str_value)
    logger.info(f"Set metadata for model {model.id}: {key}={str_value}, success={ok}")


def update_model_tags(model, add_tags=None, remove_tags=None):
    """Updates tags for a ClearML model."""
    current_tags = list(model.tags or [])

    if remove_tags:
        current_tags = [
            tag for tag in current_tags
            if tag not in remove_tags
        ]

    if add_tags:
        for tag in add_tags:
            if tag not in current_tags:
                current_tags.append(tag)

    current_tags = list(set(current_tags)) # Ensure uniqueness
    model.tags = current_tags

    logger.info(f"Updated tags for model {model.id}: {current_tags}")


def get_current_champion(project_name):
    """Queries ClearML for the current champion model."""
    try:
        # Prioritize models explicitly tagged as 'champion'
        models = Model.query_models(
            project_name=project_name,
            tags=["champion"],
            only_published=False,
            include_archived=True,
            max_results=100
        )

        if not models:
            # If no tagged champion, look for models with "role": "champion" metadata
            all_models_in_project = Model.query_models(
                project_name=project_name,
                tags=None, # Search all models in the project
                only_published=False,
                include_archived=True,
                max_results=100
            )
            champion_candidates = [
                m for m in all_models_in_project
                if get_model_metadata(m, "role") == "champion"
            ]
        else:
            champion_candidates = models

        if not champion_candidates:
            logger.info("No current champion found.")
            return None

        # Sort by creation date, newest first
        champion_candidates = sorted(
            champion_candidates,
            key=lambda m: m.created,
            reverse=True
        )

        return champion_candidates[0]

    except Exception as e:
        logger.error(f"Error fetching current champion: {e}")
        return None


def main():
    try:
        task = Task.init(
            project_name=PROJECT_NAME,
            task_name="s6_compare_with_champion"
        )
        project_name = task.project_name # Dynamically get project name
    except Exception as e:
        logger.error(f"Failed to initialize ClearML task: {e}")
        sys.exit(1)


    args = {
        "evaluation_task_id": "",
        "best_model_task_id": "" # Note: This parameter is retrieved but not used later in the script
    }
    task.connect(args)

    evaluation_task_id = task.get_parameter("General/evaluation_task_id")
    best_model_task_id = task.get_parameter("General/best_model_task_id")

    if not evaluation_task_id:
        logger.warning("Missing 'evaluation_task_id'. Exiting script.")
        return

    try:
        evaluation_task = Task.get_task(task_id=evaluation_task_id)
        if not evaluation_task:
            logger.error(f"Evaluation task with ID '{evaluation_task_id}' not found.")
            return

        if "evaluation_result" not in evaluation_task.artifacts:
            logger.error(f"Artifact 'evaluation_result' not found in evaluation task '{evaluation_task_id}'.")
            return

        challenger_result = evaluation_task.artifacts["evaluation_result"].get()

        challenger_mse = float(challenger_result.get("mse"))
        challenger_model_name = challenger_result.get("model_name")
        challenger_model_id = challenger_result.get("output_model_id")

        if not all([challenger_mse, challenger_model_name, challenger_model_id]):
            logger.error("Missing required keys (mse, model_name, output_model_id) in 'evaluation_result' artifact.")
            return

        challenger_model = Model(model_id=challenger_model_id)
        logger.info(f"Challenger Model Name: {challenger_model_name}")
        logger.info(f"Challenger Model ID: {challenger_model_id}")
        logger.info(f"Challenger MSE: {challenger_mse:.5f}")

    except Exception as e:
        logger.error(f"Error processing challenger model details: {e}")
        return

    champion_model = get_current_champion(project_name)

    champion_model_id = None
    champion_mse = None
    promote = False

    if champion_model is None:
        logger.info("No current champion found. Challenger will be promoted.")
        promote = True
    else:
        champion_model_id = champion_model.id
        # Default to infinity if MSE is not found, ensuring any valid challenger wins
        champion_mse = float(get_model_metadata(champion_model, "mse", default=float('inf')))

        logger.info(f"Current Champion Model ID: {champion_model_id}")
        logger.info(f"Current Champion MSE: {champion_mse:.5f}")

        promote = challenger_mse < champion_mse

    if promote:
        logger.info(f"Challenger (MSE: {challenger_mse:.5f}) promoted to champion.")

        if champion_model is not None:
            logger.info(f"Archiving previous champion: {champion_model.id}")
            set_model_metadata(champion_model, "role", "archived_champion")
            update_model_tags(
                champion_model,
                remove_tags=["champion"],
                add_tags=["archived_champion"]
            )

        logger.info(f"Promoting challenger model {challenger_model.id} to champion.")
        set_model_metadata(challenger_model, "role", "champion")
        set_model_metadata(challenger_model, "mse", challenger_mse)
        set_model_metadata(challenger_model, "model_name", challenger_model_name)

        update_model_tags(
            challenger_model,
            remove_tags=["rejected_challenger", "archived_champion"], # Clean up old tags
            add_tags=["champion", "challenger"]
        )
        decision = "promoted"
    else:
        logger.info(f"Current champion (MSE: {champion_mse:.5f}) remains champion. Challenger (MSE: {challenger_mse:.5f}) rejected.")

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
        "challenger_task_id": best_model_task_id, # Still unused
        "challenger_model_name": challenger_model_name,
        "challenger_mse": challenger_mse,
        "previous_champion_model_id": champion_model_id,
        "previous_champion_mse": champion_mse
    }

    try:
        task.upload_artifact(
            name="comparison_result",
            artifact_object=comparison_result,
            wait_on_upload=True
        )
        task.flush(wait_for_uploads=True)
        logger.info(f"Comparison result uploaded: {comparison_result}")
        logger.info("Process completed successfully.")
    except Exception as e:
        logger.error(f"Failed to upload artifact or flush task: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()