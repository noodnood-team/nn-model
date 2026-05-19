def build_model(model_name):
    if model_name == "resnet18":
        import torchvision.models as models
        import torch.nn as nn
        import torch

        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 4)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Model {model_name} loaded successfully.")

        return model, device
    elif model_name == "resnet34":
        import torchvision.models as models
        import torch.nn as nn
        import torch

        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 4)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Model {model_name} loaded successfully.")

        return model, device
    elif model_name == "mobilenet_v3_small":
        import torchvision.models as models
        import torch.nn as nn
        import torch
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Model {model_name} loaded successfully.")

        return model, device
    
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

def get_champion_model():
    from clearml import Task
    STEP_NAME = "s6_compare_with_champion"
    ARTIFACT_NAME = "champion_result"

    pipelines = Task.get_tasks(
        task_filter={
            "type": ["controller"],
            "status": ["completed"]
        }
    )

    if not pipelines:
        return None, None

    latest_pipeline = max(
        pipelines,
        key=lambda pipeline: pipeline.data.created
    )

    pipeline_state = latest_pipeline.get_configuration_object_as_dict(
        "Pipeline"
    )

    if STEP_NAME not in pipeline_state:
        return None, None

    step_data = pipeline_state[STEP_NAME]

    s6_task_id = (
        step_data.get("executed")
        or step_data.get("job_id")
    )

    if not s6_task_id:
        return None, None
    
    s6_task = Task.get_task(
        task_id=s6_task_id
    )

    if ARTIFACT_NAME not in s6_task.artifacts:
        return None, None

    champion_result = (
        s6_task
        .artifacts[ARTIFACT_NAME]
        .get()
    )

    model_name = champion_result.get("model_name")
    pipeline_id = champion_result.get("pipeline_id")
    model_mse = champion_result.get("mse")
    print(f"Champion model: {model_name}, from pipeline: {pipeline_id}, Model MSE: {model_mse}")

    return model_name, pipeline_id, model_mse

def get_model_weight_from_pipeline(pipeline_id):
    from clearml import Task
    import os
    import glob

    STEP_S4 = "s4_train_best_model"
    MODEL_ARTIFACT_NAME = "model"

    MODEL_DIR = "models"

    # =========================
    # Create model directory
    # =========================
    os.makedirs(MODEL_DIR, exist_ok=True)

    # =========================
    # Delete old model files
    # =========================
    old_model_files = glob.glob(
        os.path.join(MODEL_DIR, "*.pth")
    )

    for old_file in old_model_files:
        try:
            os.remove(old_file)
            print(f"Deleted old model: {old_file}")
        except Exception as e:
            print(f"Failed to delete {old_file}: {e}")

    # =========================
    # Open pipeline
    # =========================
    pipeline = Task.get_task(
        task_id=pipeline_id
    )

    pipeline_state = pipeline.get_configuration_object_as_dict(
        "Pipeline"
    )

    if not pipeline_state:
        return None

    if STEP_S4 not in pipeline_state:
        return None

    s4_step = pipeline_state[STEP_S4]

    s4_task_id = (
        s4_step.get("executed")
        or s4_step.get("job_id")
    )

    if not s4_task_id:
        return None

    # =========================
    # Open s4 task
    # =========================
    s4_task = Task.get_task(
        task_id=s4_task_id
    )

    if MODEL_ARTIFACT_NAME not in s4_task.artifacts:
        return None

    # =========================
    # Download model
    # =========================
    model_weight_path = (
        s4_task
        .artifacts[MODEL_ARTIFACT_NAME]
        .get_local_copy(
            extract_archive=False
        )
    )

    print(f"Model weight path retrieved: {model_weight_path}")

    return model_weight_path

def get_model_evaluation_from_pipeline(pipeline_id):
    from clearml import Task
    STEP_S5 = "s5_evaluate_model"
    ARTIFACT_NAME = "evaluation_result"

    # =========================
    # Open pipeline
    # =========================
    pipeline = Task.get_task(
        task_id=pipeline_id
    )

    pipeline_state = pipeline.get_configuration_object_as_dict(
        "Pipeline"
    )

    if not pipeline_state:
        return None

    # =========================
    # Find s5 step
    # =========================
    if STEP_S5 not in pipeline_state:
        return None

    s5_step = pipeline_state[STEP_S5]

    s5_task_id = (
        s5_step.get("executed")
        or s5_step.get("job_id")
    )

    if not s5_task_id:
        return None

    # =========================
    # Open s5 task
    # =========================
    s5_task = Task.get_task(
        task_id=s5_task_id
    )

    # =========================
    # Get artifact
    # =========================
    if ARTIFACT_NAME not in s5_task.artifacts:
        return None

    evaluation_result = (
        s5_task
        .artifacts[ARTIFACT_NAME]
        .get()
    )

    return evaluation_result