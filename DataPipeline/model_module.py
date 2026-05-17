def build_model(model_name):
    if model_name == "resnet18":
        import torchvision.models as models
        import torch.nn as nn
        import torch

        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 4)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        return model, device
    elif model_name == "resnet34":
        import torchvision.models as models
        import torch.nn as nn
        import torch

        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 4)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        return model, device
    elif model_name == "mobilenet_v3_small":
        import torchvision.models as models
        import torch.nn as nn
        import torch
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        return model, device
    
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

def resnet_model():
    import torchvision.models as models
    import torch.nn as nn
    import torch

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, device

def get_training_components(model, learning_rate, weight_decay):
    import torch.optim as optim
    import torch.nn as nn

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return criterion, optimizer

def train(model, train_loader, criterion, optimizer, device, num_epochs):
    from clearml import Task
    import torch
    logger = Task.current_task().get_logger()
    
    patience=10
    min_delta=0.001 

    # Initialize early stopping parameters
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        logger.report_scalar(
            title="loss",
            series="train",
            value=avg_loss,
            iteration=epoch
        )

        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # Early stopping check
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_Resnet18_model.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs: no improvement for {patience} epochs.")
            break

    return None

def save_model(model, path):
    import torch
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    return None

def eval(test_loader, model, criterion, device):
    import torch
    import numpy as np
    from clearml import Task
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    logger = Task.current_task().get_logger()

    model.eval()
    val_loss = 0

    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = val_loss / len(test_loader)

    y_pred = np.vstack(all_outputs)
    y_true = np.vstack(all_labels)

    # Reverse log transformation
    y_pred = np.expm1(y_pred)
    y_true = np.expm1(y_true)

    target_names = ["calories", "protein", "carbs", "fat"]

    metrics = {
        "mse_log_loss": float(avg_loss)
    }

    # Overall metrics
    mse_real = mean_squared_error(y_true, y_pred)
    mae_real = mean_absolute_error(y_true, y_pred)
    rmse_real = mse_real ** 0.5

    metrics["mse"] = float(mse_real)
    metrics["mae"] = float(mae_real)
    metrics["rmse"] = float(rmse_real)

    metrics["overall_mse"] = float(mse_real)
    metrics["overall_mae"] = float(mae_real)
    metrics["overall_rmse"] = float(rmse_real)

    print("MSE log loss:", avg_loss)
    print("Overall MSE:", mse_real)
    print("Overall MAE:", mae_real)
    print("Overall RMSE:", rmse_real)

    logger.report_scalar("metrics", "mse_log_loss", value=float(avg_loss), iteration=0)
    logger.report_scalar("metrics", "overall_mse", value=float(mse_real), iteration=0)
    logger.report_scalar("metrics", "overall_mae", value=float(mae_real), iteration=0)
    logger.report_scalar("metrics", "overall_rmse", value=float(rmse_real), iteration=0)
    
    logger.report_scalar("metrics", "mse", value=float(mse_real), iteration=0)
    logger.report_scalar("metrics", "mae", value=float(mae_real), iteration=0)
    logger.report_scalar("metrics", "rmse", value=float(rmse_real), iteration=0)

    # Per-target metrics
    for i, target in enumerate(target_names):
        target_y_true = y_true[:, i]
        target_y_pred = y_pred[:, i]

        target_mse = mean_squared_error(target_y_true, target_y_pred)
        target_mae = mean_absolute_error(target_y_true, target_y_pred)
        target_rmse = target_mse ** 0.5

        metrics[f"{target}_mse"] = float(target_mse)
        metrics[f"{target}_mae"] = float(target_mae)
        metrics[f"{target}_rmse"] = float(target_rmse)

        print(f"{target} MSE:", target_mse)
        print(f"{target} MAE:", target_mae)
        print(f"{target} RMSE:", target_rmse)

        logger.report_scalar("metrics", f"{target}_mse", value=float(target_mse), iteration=0)
        logger.report_scalar("metrics", f"{target}_mae", value=float(target_mae), iteration=0)
        logger.report_scalar("metrics", f"{target}_rmse", value=float(target_rmse), iteration=0)

    return metrics