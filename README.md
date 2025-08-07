
# 🧠 Loblaw Budget Optimizer

Welcome to **Loblaw Budget Optimizer** — a collaborative, cloud-ready machine learning pipeline that recommends weekly ad budget allocations across 8 digital channels.

Built with ❤️ using PyTorch, Transformers, and a touch of S3 magic.

---

## 🚀 What It Does

This project simulates and models a multi-week advertising campaign. It:

- Dynamically recommends weekly budget allocations `{channel_name: %}`  
- Trains a Transformer-based Allocator + ROAS Predictor  
- Learns from historical campaign data (real or mock)  
- Outputs recommendations via an `InferencePipeline`  
- Is production-ready with support for AWS S3 and SageMaker

---

## 🧱 Project Structure

```
.
├── src/                      # Core ML code
│   ├── dataloader.py         # Sequence-aware Dataset class
│   ├── model.py              # Allocator and Predictor networks
│   ├── trans_encoder.py      # Transformer encoder with positional encoding
│   ├── training.py           # Two-stage training pipeline
│   ├── pipeline.py           # Inference-ready budget allocation tool
│
├── orchestration/
│   └── config/
│       └── config.py         # Hyperparams, paths, S3 toggles
│
├── data/
│   └── mock_campaign_data.csv  # Simulated dataset
│
├── notebook/
│   └── demo.ipynb            # Jupyter demo notebook (optional)
│
└── README.md                 # You’re reading this
```

---

## ⚙️ Setup Instructions

### 1. Clone the repo and install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configure parameters in:

```python
orchestration/config/config.py
```

Set paths like:
```python
"csv_data_path": "s3://loblaw-data/input/mock_campaign_data.csv"
"upload_s3_path": "s3://loblaw-models/budget/model.pt"
```

Or run locally:
```python
"csv_data_path": "data/mock_campaign_data.csv"
"upload_s3_path": ""
```

---

## 🧪 Training

Run the two-phase training loop (predictor + allocator):

```bash
python src/training.py
```

Outputs:
- Trained model weights (`.pt`)
- Optional upload to S3 if configured

---

## 🤖 Inference

After training, generate weekly allocation recommendations:

```python
from src.pipeline import InferencePipeline

pipeline = InferencePipeline()
result = pipeline.predict(input_sequence)  # ← pass 8-week input as list of dicts
```

Returns:
```python
{
  "Meta": 0.21,
  "Google": 0.18,
  "YouTube": 0.22,
  ...
}
```

---

## ☁️ Cloud Compatibility (AWS)

- ✅ Reads/writes from Amazon S3
- ✅ Deployable in SageMaker (training + endpoint)
- ✅ Modular for use in Step Functions or Lambda
- ✅ Model artifacts stored in S3 with versioning

---

## 🧠 Model Highlights

- Transformer-based encoder for time-series campaigns
- Allocator (budget planner) + Predictor (ROAS forecaster)
- Two-stage training loop for collaboration between models
- Sequence padding, cumulative allocation, and more

---

## 🙌 Team-Friendly Design

This project was built with collaboration in mind:
- Easy-to-edit configs
- Modular components
- Reproducible training + inference
- Notebook optional, but encouraged 🎓

---

## 🧊 Cool Bonus Ideas (Optional)

- Add a forecasting module for spend/revenue per allocation
- Plug into Streamlit or Flask for live demo
- Visualize pacing trends in QuickSight

---

## 📬 Questions?

Feel free to ping me if you’re curious about:
- Model logic
- Data generation
- SageMaker deployment
- Or just want to nerd out about bandits vs. transformers 😄
