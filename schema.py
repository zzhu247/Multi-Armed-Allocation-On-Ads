import pandas as pd

dtypes = {
    "week": "int64",
    "channel": "string",
    "spend": "float64",
    "impressions": "int64",
    "clicks": "int64",
    "conversions": "int64",
    "revenue": "float64"
}

df = pd.read_csv("s3://your-bucket/your-campaign-data.csv", dtype=dtypes)
