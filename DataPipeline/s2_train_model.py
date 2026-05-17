from clearml import Task
import pandas as pd
import logging

from data_module import create_dataloader
from model_module import build_model, get_training_components, train, save_model, eval

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
# Initialize the task
    task = Task.init(
        project_name='NutritionAnalyser',
        task_name='s2_train_model',
    )

    args = {
        "preprocess_task_id": "",
        "model_name": "resnet18", # will be replaced by parameter in pipeline
        'num_epochs': 20, # will be replaced by parameter in pipeline
        'batch_size': 32, # will be replaced by parameter in pipeline
        'learning_rate': 1e-3, # will be replaced by parameter in pipeline
        'weight_decay': 1e-5 # will be replaced by parameter in pipeline
    }
    task.connect(args)
    logger.info(f"Connected parameters: {args}")

    preprocess_task_id = task.get_parameter("General/preprocess_task_id")

    # Check if preprocess_task_id is provided, if not, exit the function (this allows us to create a base task template without running the full training)
    if not preprocess_task_id:
        logger.info("No preprocess_task_id provided. This run is only for creating a base task template.")
        return

    s1_task = Task.get_task(task_id=preprocess_task_id)

    # load train_df and test_df from input artifacts
    train_df_path = s1_task.artifacts["train_df"].get_local_copy()
    test_df_path = s1_task.artifacts["test_df"].get_local_copy()
    train_df = pd.read_csv(train_df_path)
    test_df = pd.read_csv(test_df_path)
    logger.info("Finished loading preprocessed data.")

    # create dataloader
    train_loader, test_loader = create_dataloader(train_df, test_df, batch_size=args['batch_size'])
    logger.info("Created dataloaders for training and testing.")

    # initialize model and training components
    model, device = build_model(args["model_name"])
    criterion, optimizer = get_training_components(model, args['learning_rate'], args['weight_decay'])

    # train the model
    train(model, train_loader, criterion, optimizer, device, num_epochs=args['num_epochs'])
    logger.info("Finished training the model.")

    # Save the model
    model_path = f"artifacts/model/{args['model_name']}.pth"
    save_model(model, model_path)
    logger.info("Saved the trained model")

    # Evaluate the model on the test set and log the results
    evaluation_result = eval(test_loader, model, criterion, device)
    mse = float(evaluation_result["mse"])

    task.get_logger().report_scalar(
        title="metrics",
        series="mse",
        value=mse,
        iteration=args["num_epochs"]
    )

    task.set_parameter("General/mse", mse)
    logger.info(f"Training evaluation MSE: {mse}")

    # Upload model artifact
    task.upload_artifact(
        name="model",
        artifact_object=model_path,
        wait_on_upload=True
    )
    
    # Upload model metadata as an artifact (including hyperparameters and model path)
    task.upload_artifact(
        name="model_metadata",
         artifact_object={
            "model_name": args["model_name"],
            "num_epochs": args["num_epochs"],
            "batch_size": args["batch_size"],
            "learning_rate": args["learning_rate"],
            "weight_decay": args["weight_decay"],
            "model_path": model_path
        },
        wait_on_upload=True
    )

    logger.info(f"s2_train_model completed for model: {args['model_name']}")

if __name__ == "__main__":
    main()