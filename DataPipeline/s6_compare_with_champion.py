from clearml import Task, Model
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_NAME = "NutritionAnalyser"


# get champion model details from previous pipeline step s6_compare_with_champion.py
def get_champion_pipeline_details():
    pipelines = Task.get_tasks(
        task_filter={
            "type": ["controller"],
            "status": ["completed"],
        }
    )

    if not pipelines:
        logger.info("No completed pipelines found.")
        return None, None, None
    else:
        latest_pipeline = max(
            pipelines,
            key=lambda x: x.data.created
        )
        pipeline_state = latest_pipeline.get_configuration_object_as_dict("Pipeline")

        if "s6_compare_with_champion" not in pipeline_state:
            logger.info("No s6_compare_with_champion step found in the latest pipeline.")
            return None, None, None
        
        step_data = pipeline_state["s6_compare_with_champion"]
        executed_task_id = step_data.get("executed") or step_data.get("job_id")
        step_task = Task.get_task(task_id=executed_task_id)
        champion_pipeline_id = step_task.get_parameter("General/champion_pipeline_id")
        champion_model_name = step_task.get_parameter("General/champion_model_name")
        champion_mse = float(step_task.get_parameter("General/champion_mse"))

        return champion_pipeline_id, champion_model_name, champion_mse
    
def find_pipeline_by_running_task_id(running_task_id):
    pipelines = Task.get_tasks(
        task_filter={
            "type": ["controller"]
        }
    )

    for pipeline in pipelines:
        pipeline_state = pipeline.get_configuration_object_as_dict("Pipeline")

        if not pipeline_state:
            continue

        for step_name, step_data in pipeline_state.items():
            executed_task_id = (
                step_data.get("executed")
                or step_data.get("job_id")
            )

            if executed_task_id == running_task_id:
                return pipeline.id

    return None

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

    challenger_task_id = task.id
    challenger_mse = float(challenger_result["mse"])
    challenger_model_name = challenger_result["model_name"]

    challenger_pipeline_id = find_pipeline_by_running_task_id(challenger_task_id)
    champion_pipeline_id, champion_model_name, champion_mse = get_champion_pipeline_details()
    
    logger.info(f"challenger_pipeline_id: {challenger_pipeline_id}")
    logger.info(f"champion_pipeline_id: {champion_pipeline_id}")

    if champion_pipeline_id is None:
        logger.info("No champion pipeline found. Challenger becomes champion by default.")
        champion_result = {
            "pipeline_id": challenger_pipeline_id,
            "model_name": challenger_model_name,
            "mse": challenger_mse
        }
    else:
        logger.info(f"Challenger MSE: {challenger_mse}")
        logger.info(f"Champion MSE: {champion_mse}")

        if challenger_mse < champion_mse:
            logger.info("Challenger outperforms champion! Updating champion details.")
            champion_result = {
                "pipeline_id": challenger_pipeline_id,
                "model_name": challenger_model_name,
                "mse": challenger_mse
            }
        else:
            champion_result = {
                "pipeline_id": champion_pipeline_id,
                "model_name": champion_model_name,
                "mse": champion_mse
            }
    
    task.set_parameter("General/champion_pipeline_id", champion_result["pipeline_id"])
    task.set_parameter("General/champion_model_name", champion_result["model_name"])
    task.set_parameter("General/champion_mse", str(champion_result["mse"]))

    task.upload_artifact(
        name="champion_result",
        artifact_object=champion_result,
        wait_on_upload=True
    )

    task.flush(wait_for_uploads=True)

    logger.info(
        "s6_compare_with_champion completed."
    )


if __name__ == "__main__":
    main()
