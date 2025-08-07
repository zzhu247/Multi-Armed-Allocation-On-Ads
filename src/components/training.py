import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from orchestration.config.config import config
from dataloader import CampaignDataset
from trans_encoder import TransformerEncoder
from model import Predictor, Allocator

import boto3

def upload_to_s3(local_path, s3_path):
    """
    Uploads a local file to the given S3 path.
    """
    s3 = boto3.client('s3')
    bucket, key = s3_path.replace("s3://", "").split("/", 1)
    s3.upload_file(local_path, bucket, key)
    print(f"Uploaded {local_path} to {s3_path}")

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = CampaignDataset(config["csv_data_path"], config)
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

# Initialize models
encoder = TransformerEncoder(
    input_dim=len(config["input_columns"]) + 8,
    hidden_dim=config["hidden_dim"],
    num_heads=config["num_heads"],
    num_layers=config["num_layers"],
    dropout=config["dropout"]
).to(device)

predictor = Predictor(config["hidden_dim"], alloc_dim=8).to(device)
allocator = Allocator(config["hidden_dim"], alloc_dim=8).to(device)

# Optimizers
optimizer_pred = optim.Adam(predictor.parameters(), lr=config["lr"])
optimizer_alloc = optim.Adam(allocator.parameters(), lr=config["lr"])

# Loss
mse = nn.MSELoss()

# --- Training loop ---
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

        # === Stage 1: Train predictor ===
        h = encoder(x)  # (batch, hidden)
        pred_roas = predictor(h, real_alloc)
        loss_pred = mse(pred_roas, roas)

        optimizer_pred.zero_grad()
        loss_pred.backward()
        optimizer_pred.step()

        total_pred_loss += loss_pred.item() * x.size(0)

        # === Stage 2: Train allocator (freeze predictor) ===
        with torch.no_grad():
            h = encoder(x)
            roas_true = roas.detach()

        r_prime = allocator(h)
        improved_roas = predictor(h.detach(), r_prime)
        loss_alloc = - (improved_roas - roas_true).mean()

        optimizer_alloc.zero_grad()
        loss_alloc.backward()
        optimizer_alloc.step()

        total_alloc_gain += -loss_alloc.item() * x.size(0)

    avg_pred_loss = total_pred_loss / len(dataset)
    avg_alloc_gain = total_alloc_gain / len(dataset)

    print(f"Epoch {epoch+1}/{config['num_epochs']} | ROAS MSE: {avg_pred_loss:.4f} | Allocator Gain: {avg_alloc_gain:.4f}")

# Save models
# Save locally
encoder_path = config["model_save_path"].replace(".pt", "_encoder.pt")
predictor_path = config["model_save_path"].replace(".pt", "_predictor.pt")
allocator_path = config["model_save_path"].replace(".pt", "_allocator.pt")

torch.save(encoder.state_dict(), encoder_path)
torch.save(predictor.state_dict(), predictor_path)
torch.save(allocator.state_dict(), allocator_path)

print("Saved models locally.")

# Upload to S3 if config["upload_s3_path"] is provided
if "upload_s3_path" in config and config["upload_s3_path"]:
    upload_to_s3(encoder_path, config["upload_s3_path"].replace(".pt", "_encoder.pt"))
    upload_to_s3(predictor_path, config["upload_s3_path"].replace(".pt", "_predictor.pt"))
    upload_to_s3(allocator_path, config["upload_s3_path"].replace(".pt", "_allocator.pt"))

print(f"Saved encoder, predictor, allocator models to: {config['model_save_path']}")
