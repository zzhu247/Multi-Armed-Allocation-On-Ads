config = {
    # Hyperparameters
    "hidden_dim": 128,
    "num_heads": 4,
    "num_layers": 2,
    "dropout": 0.1,
    "batch_size": 32,
    "lr": 1e-3,
    "num_epochs": 10,
    "seq_len": 5,

    # File paths
    "model_save_path": "actor_model.pt",
    "csv_data_path": "s3://loblaw-data/input/mock_campaign_data.csv",

    # Input feature columns (ordered)
    "input_columns": [
        "RCS_clicks", "RCS_impressions", "RCS_completion_rate", "RCS_allocated_pct",
        "Shoppers_clicks", "Shoppers_impressions", "Shoppers_completion_rate", "Shoppers_allocated_pct",
        "Meta_clicks", "Meta_impressions", "Meta_completion_rate", "Meta_allocated_pct",
        "Snap_clicks", "Snap_impressions", "Snap_completion_rate", "Snap_allocated_pct",
        "YouTube_clicks", "YouTube_impressions", "YouTube_completion_rate", "YouTube_allocated_pct",
        "TikTok_clicks", "TikTok_impressions", "TikTok_completion_rate", "TikTok_allocated_pct",
        "Google_clicks", "Google_impressions", "Google_completion_rate", "Google_allocated_pct",
        "Bing_clicks", "Bing_impressions", "Bing_completion_rate", "Bing_allocated_pct"
    ],

    # Target (ground truth) column
    "target_column": "roas",

    # Output channel order
    "output_channels": [
        "RCS", "Shoppers", "Meta", "Snap",
        "YouTube", "TikTok", "Google", "Bing"
    ]
}
