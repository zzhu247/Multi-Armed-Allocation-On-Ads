import torch
import numpy as np
from orchestration.config.config import config
from components.trans_encoder import TransformerEncoder
from components.model import Allocator

class InferencePipeline:
    def __init__(self):
        input_dim = len(config["input_columns"]) + 8
        hidden_dim = config["hidden_dim"]
        num_channels = 8

        # Load encoder
        self.encoder = TransformerEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            dropout=config["dropout"]
        )
        self.encoder.load_state_dict(torch.load(config["model_save_path"].replace(".pt", "_encoder.pt"), map_location="cpu"))
        self.encoder.eval()

        # Load allocator
        self.allocator = Allocator(hidden_dim, alloc_dim=num_channels)
        self.allocator.load_state_dict(torch.load(config["model_save_path"].replace(".pt", "_allocator.pt"), map_location="cpu"))
        self.allocator.eval()

        self.device = torch.device("cpu")
        self.encoder.to(self.device)
        self.allocator.to(self.device)

        # Cached config
        self.input_columns = config["input_columns"]
        self.output_channels = config["output_channels"]
        self.seq_len = config["seq_len"]
        self.full_feature_order = self.input_columns + [
            col.replace("allocated_pct", "accum_allocated_pct")
            for col in self.input_columns if col.endswith("allocated_pct")
        ]

    def predict(self, input_sequence: list[dict]) -> dict:
        """
        input_sequence: list of dicts, each dict has 40 keys
        Returns: dict with 8 allocations, keyed by channel name
        """
        if len(input_sequence) != self.seq_len:
            raise ValueError(f"Expected sequence of length {self.seq_len}, got {len(input_sequence)}.")

        for i, row in enumerate(input_sequence):
            missing = set(self.full_feature_order) - set(row)
            if missing:
                raise ValueError(f"Missing features in timestep {i}: {missing}")

        # Convert to ordered tensor
        feature_matrix = [
            [row[col] for col in self.full_feature_order]
            for row in input_sequence
        ]
        input_tensor = torch.tensor([feature_matrix], dtype=torch.float32).to(self.device)  # (1, seq_len, input_dim)

        with torch.no_grad():
            h = self.encoder(input_tensor)  # (1, hidden_dim)
            alloc_vector = self.allocator(h)  # (1, 8)

        alloc_vector = alloc_vector.squeeze(0).tolist()
        return {channel: round(percent, 2) for channel, percent in zip(self.output_channels, alloc_vector)}
