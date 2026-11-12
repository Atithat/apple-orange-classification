# 🍎 FruitLens — Apple vs Orange CNN Classification

Streamlit web app + trainable TensorFlow CNN for binary **apple/orange image classification**, prepared for the KMITL Machine Learning classification project due **20 November 2026**.

> **Status:** Source code uploaded. A trained model and results are not provided until `python train_model.py` has been run. This repo does not claim any unmeasured accuracy.

## Run locally (Windows)

Install Python 3.11, clone/download this repository, and run in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```

Your browser will open **http://localhost:8501**. On future runs, start Streamlit without retraining.

You can also double-click **run_windows.bat** to set up the environment automatically.

## CNN / rubric

| Item | Implementation |
|---|---|
| Task | Apple (label 0) / Orange (label 1) |
| Dataset | [Fruits-360 100x100](https://github.com/fruits-360/fruits-360-100x100) |
| Images | 180 train + 48 validation + 32 held-out tests |
| Validation percentage | 48 / 180 = **26.7%** (at least 20% required) |
| Input | RGB 128 × 128 |
| Layers | Conv2D 24, 48, 96 + pooling; Dense 64; Sigmoid output |
| Optimizer / loss | Adam (0.001), binary cross-entropy |
| Training | Up to 12 epochs, early stopping, batch 16 |
| Test | Held-out official Fruits-360 Test partition + four distinct examples |
| App | Image upload, camera, class probabilities, training graphs |
| Report | Introduction, Methodology, Train and Test, Conclusion, Further Work |

The selected dataset folders use multiple apple varieties and orange varieties; downloaded images are cached under `data/`. Downloading requires an internet connection and could be affected by GitHub rate limits.

## Training outputs

```text
models/apple_orange_cnn.keras           # trained CNN
outputs/evaluation.json                # measured metrics
outputs/figures/training_history.png    # learning curves
outputs/figures/confusion_matrix.png   # test confusion matrix
outputs/figures/four_test_predictions.png
examples/test_*.jpg                    # four held-out demonstration images
```

Your professor also requires a screenshot of the working website with its **URL bar visible**. Capture a real screenshot from localhost or Streamlit Community Cloud (not a fabricated screenshot), then add it to your final report.

## Deploy on Streamlit Community Cloud

1. Train the model locally first.
2. Commit or upload the resulting `models/`, `examples/`, and `outputs/` files to this repository. Do not upload `data/` (raw dataset).
3. Go to https://share.streamlit.io/ and choose **Create app**.
4. Select this repository, branch `main`, entrypoint `app.py`, and Python 3.11.
5. Once deployed, verify both fruit predictions and include the resulting public URL in the report.

**Important:** Just uploading Python source does not make the deployed classifier ready. The app intentionally displays setup instructions until the trained `.keras` model exists.

## Project files

- `app.py` — Streamlit website
- `train_model.py` — end-to-end CNN training, evaluations and plots
- `dataset.py` — reproducible Fruits-360 image acquisition and split construction
- `model.py` — Keras CNN architecture
- `inference.py` — shared preprocessing and model prediction
- `requirements.txt` — dependencies
- `run_windows.bat` — Windows setup launcher

### Attribution

Fruit images come from [Fruits-360](https://github.com/fruits-360/fruits-360-100x100), originally created by Mihai Oltean and collaborators and shared under **CC BY-SA 4.0**. This project uses only a small sample and attributes the source; check attribution/share-alike obligations if redistributing image files.

### Evaluation caveat

Fruits-360 test examples are often visually similar to training shots (plain backgrounds / nearby rotations). Test accuracy on those images does not guarantee accurate predictions on cluttered phone photographs. The binary model has no unknown-object class.
