import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

import pandas as pd
import torch
from torch.utils.data import Dataset

import boto3
import io
import pandas as pd

def load_csv_from_s3(s3_path):
    s3 = boto3.client('s3')
    bucket, key = s3_path.replace("s3://", "").split("/", 1)
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj['Body'].read()))


class CampaignDataset(Dataset):
    def __init__(self, csv_path, config):
        self.seq_len = config["seq_len"]
        self.input_cols = config["input_columns"]
        self.target_col = config["target_column"]
        self.alloc_cols = [col for col in self.input_cols if col.endswith("allocated_pct")]
        # load CSV data from S3
        if csv_path.startswith("s3://"):
            self.df = load_csv_from_s3(csv_path)
        else:
            self.df = pd.read_csv(csv_path)


        self.samples = []

        for _, campaign_df in self.df.groupby("id"):
            campaign_df = campaign_df.sort_values("week_number").reset_index(drop=True)

            # Compute cumulative allocation per channel
            for col in self.alloc_cols:
                cum_col = col.replace("allocated_pct", "accum_allocated_pct")
                campaign_df[cum_col] = campaign_df[col].cumsum()

            # Combine original + cumulative feature names
            cum_cols = [col.replace("allocated_pct", "accum_allocated_pct") for col in self.alloc_cols]
            all_feature_cols = self.input_cols + cum_cols

            # Generate sequence samples
            for i in range(len(campaign_df)):
                end = i + 1
                start = max(0, end - self.seq_len)
                seq = campaign_df.iloc[start:end]

                pad_len = self.seq_len - len(seq)
                if pad_len > 0:
                    pad = pd.DataFrame(0, index=range(pad_len), columns=seq.columns)
                    seq = pd.concat([pad, seq], ignore_index=True)

                x = seq[all_feature_cols].values.astype("float32")
                y = campaign_df.iloc[i][self.target_col]
                real_alloc = campaign_df.iloc[i][self.alloc_cols].values.astype("float32")

                self.samples.append((x, real_alloc, y))

        self.feature_dim = len(all_feature_cols)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, real_alloc, y = self.samples[idx]
        return (
            torch.tensor(x),                      # (seq_len, input_dim)
            torch.tensor(real_alloc),             # (8,)
            torch.tensor(y, dtype=torch.float32)  # scalar ROAS
        )
