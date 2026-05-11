"""
This module provides a function to load the nutrition dataset from ClearML. 
It retrieves the dataset using the ClearML Dataset API and returns the local path where the dataset is stored. 
The function also includes logging to track the loading process and handles exceptions that may occur during dataset retrieval.
"""
def load_data():
    from clearml import Task, Dataset
    import logging
    logger = logging.getLogger(__name__)
    logger.info("start loading dataset from ClearML...")

    try:
        dataset = Dataset.get(
            dataset_project="NutritionAnalyser",
            dataset_name="nutrition5k_dataset"
        )

        local_path = dataset.get_local_copy()

        logger.info(f"Loaded dataset name: {dataset.name}")
        logger.info(f"Dataset ID: {dataset.id}")
        logger.info(f"Dataset local path: {local_path}")
        logger.info("Finished loading dataset.")

        return local_path

    except Exception as e:
        raise RuntimeError(f"Error loading dataset: {e}")
    
"""

"""
def parse_basic(path):
    import pandas as pd
    rows = []

    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")

            try:
                rows.append({
                    "dish_id": parts[0].replace("dish_", ""),
                    "total_calories": float(parts[1]),
                    "total_mass": float(parts[2]),
                    "total_fat": float(parts[3]),
                    "total_carb": float(parts[4]),
                    "total_protein": float(parts[5]),
                })
            except:
                continue

    return pd.DataFrame(rows)

"""
"""
def clean_data(df):
    # remove rows with zero calories
    df_nozero = df[df["total_calories"] > 0]

    return df_nozero

"""
"""
def preprocessing(local_path):
    import logging
    import os
    import pandas as pd

    logger = logging.getLogger(__name__)
    logger.info("start preprocessing dataset...")

    df1 = parse_basic(os.path.join(local_path, "dish_metadata_cafe1.csv"))
    df2 = parse_basic(os.path.join(local_path, "dish_metadata_cafe2.csv"))

    df = pd.concat([df1, df2], ignore_index=True)

    # map image path to each dish
    df["image_path"] = df["dish_id"].apply(
        lambda x: os.path.join(local_path, f"dish_{x}_rgb.png")
    )

    df = df[df["image_path"].apply(os.path.exists)]

    # remove rows with zero calories    
    df = clean_data(df)

    logger.info("Preprocessed dataset with {} samples".format(len(df)))
    return df

"""
"""
def feature_engineering(df):
    import numpy as np
    # log-transform nutritional values to reduce skewness in the distribution and make it easier for the model to learn
    df["calories_log"] = np.log1p(df["total_calories"])
    df["protein_log"] = np.log1p(df["total_protein"])
    df["carb_log"] = np.log1p(df["total_carb"])
    df["fat_log"] = np.log1p(df["total_fat"])

    return df

"""
"""
def split_data(df, test_size, random_state):
    from sklearn.model_selection import train_test_split
    import pandas as pd

    df["cal_bin"] = pd.qcut(df["total_calories"], q=5, labels=False) # create 5 bins of calories for stratified splitting

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df["cal_bin"]) # stratify by calorie bins to ensure similar distribution of calories in train and test sets
    train_df = train_df.drop(columns=["cal_bin"]).reset_index(drop=True)
    test_df = test_df.drop(columns=["cal_bin"]).reset_index(drop=True)

    # select only the columns we need for training the model
    train_df = train_df[[
        "image_path",
        "calories_log",
        "protein_log",
        "carb_log",
        "fat_log"
    ]]

    test_df = test_df[[
        "image_path",
        "calories_log",
        "protein_log",
        "carb_log",
        "fat_log"
    ]]

    return train_df, test_df

"""
FoodDataset class and dataloader creation function for training the model.
"""
from clearml import Dataset
from PIL import Image
import torch
class FoodDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row["image_path"]).convert("RGB")
        except:
            return self.__getitem__((idx + 1) % len(self.df))

        if self.transform:
            img = self.transform(img)

        label = torch.tensor([
            row["calories_log"],
            row["protein_log"],
            row["carb_log"],
            row["fat_log"]
        ], dtype=torch.float32)
        
        return img, label

"""
"""
def create_dataloader(train_df, test_df, batch_size):
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_loader = DataLoader(
        FoodDataset(train_df, train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        FoodDataset(test_df, test_transform),
        batch_size=batch_size,
        num_workers=4
    )

    return train_loader, test_loader