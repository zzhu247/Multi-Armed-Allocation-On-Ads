import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import sys
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from orchestration.config.config import config
from dataloader import CampaignDataset
from trans_encoder import TransformerEncoder
from model import Predictor, Allocator

# Step 1: Load from Delta Lake
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
df = spark.read.table("criteo_budget_allocation").toPandas()

# Step 2: Clean and normalize
df = df.dropna()

roas_cols = [c for c in df.columns if c.endswith("_ROAS")]
roas_cols.append("roas")
df[roas_cols] = MinMaxScaler().fit_transform(df[roas_cols])

scale_cols = [c for c in df.columns if c.endswith("_clicks") or c.endswith("_impressions")]
df[scale_cols] = MinMaxScaler().fit_transform(df[scale_cols])

normalized_path = "/tmp/criteo_normalized.csv"
df.to_csv(normalized_path, index=False)
config["csv_data_path"] = normalized_path

# Step 3: Train with MLflow
mlflow.set_experiment("/Users/zzhuliz@outlook.com/budget-allocation-transformer")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = CampaignDataset(config["csv_data_path"], config)
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

with mlflow.start_run(run_name="criteo-v1"):

    mlflow.log_params({
        "hidden_dim": config["hidden_dim"],
        "num_heads": config["num_heads"],
        "num_layers": config["num_layers"],
        "dropout": config["dropout"],
        "batch_size": config["batch_size"],
        "lr": config["lr"],
        "num_epochs": config["num_epochs"],
        "seq_len": config["seq_len"],
        "dataset": "criteo_normalized"
    })

    encoder = TransformerEncoder(
        input_dim=len(config["input_columns"]) + 8,
        hidden_dim=config["hidden_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout=config["dropout"]
    ).to(device)

    predictor = Predictor(config["hidden_dim"], alloc_dim=8).to(device)
    allocator = Allocator(config["hidden_dim"], alloc_dim=8).to(device)

    optimizer_pred = optim.Adam(predictor.parameters(), lr=config["lr"])
    optimizer_alloc = optim.Adam(allocator.parameters(), lr=config["lr"])
    mse = nn.MSELoss()

    for epoch in range(config["num_epochs"]):
        encoder.train()
        predictor.train()
        allocator.train()

        total_pred_loss = 0.0
        total_alloc_gain = 0.0

        for x, real_alloc, roas in dataloader:
            x = x.to(device)
            real_alloc = real_alloc.to(device)
            roas = roas.to(device)

            h = encoder(x)
            pred_roas = predictor(h, real_alloc)
            loss_pred = mse(pred_roas, roas)
            optimizer_pred.zero_grad()
            loss_pred.backward()
            optimizer_pred.step()
            total_pred_loss += loss_pred.item() * x.size(0)

            with torch.no_grad():
                h = encoder(x)
                roas_true = roas.detach()

            r_prime = allocator(h)
            improved_roas = predictor(h.detach(), r_prime)
            loss_alloc = -(improved_roas - roas_true).mean()
            optimizer_alloc.zero_grad()
            loss_alloc.backward()
            optimizer_alloc.step()
            total_alloc_gain += -loss_alloc.item() * x.size(0)

        avg_pred_loss = total_pred_loss / len(dataset)
        avg_alloc_gain = total_alloc_gain / len(dataset)

        mlflow.log_metrics({
            "roas_mse": avg_pred_loss,
            "allocator_gain": avg_alloc_gain
        }, step=epoch)

        print(f"Epoch {epoch+1}/{config['num_epochs']} | ROAS MSE: {avg_pred_loss:.4f} | Allocator Gain: {avg_alloc_gain:.4f}")

    mlflow.pytorch.log_model(encoder, "encoder")
    mlflow.pytorch.log_model(predictor, "predictor")
    mlflow.pytorch.log_model(allocator, "allocator")

    print("Training complete!")
