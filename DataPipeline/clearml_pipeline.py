"""
This code defines a ClearML pipeline for the NutritionAnalyser project.

MLOps Level 2 updates:
- Baseline model training
- Hyperparameter optimization
- Train best model from HPO result
- Final model evaluation
- Champion challenger comparison
"""

print("Starting ClearML pipeline...")

from clearml import PipelineController
import logging


logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


PROJECT_NAME = "NutritionAnalyser"
SERVICE_QUEUE = "hpo_service"
TRAINING_QUEUE = "default"



pipe = PipelineController(
    name="NutritionAnalyser Pipeline",
    project=PROJECT_NAME,
    version="2.0",
)

pipe.set_default_execution_queue(TRAINING_QUEUE)

pipe.add_step(
    name="s1_data_preprocessing",
    base_task_project=PROJECT_NAME,
    base_task_name="s1_data_preprocessing",
    execution_queue=TRAINING_QUEUE,
    parameter_override={
        "General/test_size": 0.25,
        "General/random_state": 42
    }
)

pipe.add_step(
    name="s2_train_model",
    parents=["s1_data_preprocessing"],
    base_task_project=PROJECT_NAME,
    base_task_name="s2_train_model",
    execution_queue=TRAINING_QUEUE,
    parameter_override={
        "General/preprocess_task_id": "${s1_data_preprocessing.id}",
        "General/model_name": "resnet18",
        "General/num_epochs": 5,
        "General/batch_size": 32,
        "General/learning_rate": 1e-3,
        "General/weight_decay": 1e-5
    }
)


pipe.add_step(
    name="s3_hpo",
    parents=["s1_data_preprocessing", "s2_train_model"],
    base_task_project=PROJECT_NAME,
    base_task_name="s3_hpo",
    execution_queue=SERVICE_QUEUE,
    parameter_override={
        "General/preprocess_task_id": "${s1_data_preprocessing.id}",
        "General/base_train_task_id": "${s2_train_model.id}",
        "General/max_number_of_experiments": 1 ,
        "General/max_concurrent_tasks": 1,
        "General/execution_queue": TRAINING_QUEUE
    }
)


pipe.add_step(
    name="s4_train_best_model",
    parents=["s3_hpo"],
    base_task_project=PROJECT_NAME,
    base_task_name="s4_train_best_model",
    execution_queue=TRAINING_QUEUE,
    parameter_override={
        "General/preprocess_task_id": "${s1_data_preprocessing.id}",
        "General/hpo_task_id": "${s3_hpo.id}",
        "General/num_epochs": 10
    }
)


pipe.add_step(
    name="s5_evaluate_model",
    parents=["s4_train_best_model"],
    base_task_project=PROJECT_NAME,
    base_task_name="s5_evaluate_model",
    execution_queue=TRAINING_QUEUE,
    parameter_override={
        "General/preprocess_task_id": "${s1_data_preprocessing.id}",
        "General/best_model_task_id": "${s4_train_best_model.id}",
        "General/batch_size": 32
    }
)

pipe.add_step(
    name="s6_compare_with_champion",
    parents=["s5_evaluate_model"],
    base_task_project=PROJECT_NAME,
    base_task_name="s6_compare_with_champion",
    execution_queue=TRAINING_QUEUE,
    parameter_override={
        "General/evaluation_task_id": "${s5_evaluate_model.id}",
        "General/best_model_task_id": "${s4_train_best_model.id}"
    }
)


logger.info("Starting pipeline locally with tasks on queue: default")
pipe.start_locally()
logger.info("Pipeline started successfully")