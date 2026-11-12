"""Download fruit pictures, train a CNN, evaluate and export real results."""
from pathlib import Path
import argparse
import json
import random
import shutil

import numpy as np
from PIL import Image
from dataset import prepare_dataset, SEED
from model import build_model, IMAGE_SIZE
from inference import predict_fruit

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODEL = ROOT / "models" / "apple_orange_cnn.keras"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
EXAMPLES = ROOT / "examples"

def save_figures(history, confusion, examples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    xs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key in zip(axes, ["accuracy", "loss"]):
        ax.plot(xs, history[key], "o-", label="Training")
        ax.plot(xs, history["val_" + key], "o-", label="Validation")
        ax.set(title=key.title(), xlabel="Epoch", ylabel=key.title())
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "training_history.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(confusion, cmap="Blues")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(confusion[row, col]), ha="center", va="center", fontsize=14)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Apple", "Orange"],
           yticklabels=["Apple", "Orange"], xlabel="Predicted", ylabel="Actual",
           title="Held-out confusion matrix")
    fig.tight_layout()
    fig.savefig(FIG / "confusion_matrix.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    for ax, item in zip(axes, examples):
        ax.imshow(Image.open(ROOT / item["image_path"]))
        ax.set_title("True: %s\nPred: %s (%.1f%%)" % (
            item["expected_label"], item["predicted_label"], item["confidence"] * 100), fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "four_test_predictions.png", dpi=170)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--train-per-class", type=int, default=90)
    parser.add_argument("--val-per-class", type=int, default=24)
    parser.add_argument("--test-per-class", type=int, default=16)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")

    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    manifest = prepare_dataset(DATA, args.train_per_class, args.val_per_class, args.test_per_class)

    def load(split, shuffle=False):
        return tf.keras.utils.image_dataset_from_directory(
            DATA / split, labels="inferred", label_mode="binary",
            class_names=["apple", "orange"], image_size=IMAGE_SIZE,
            batch_size=16, shuffle=shuffle, seed=SEED
        ).prefetch(tf.data.AUTOTUNE)

    model = build_model()
    model.summary()
    training = model.fit(
        load("train", shuffle=True), validation_data=load("val"), epochs=args.epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5)
        ], verbose=2
    )
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL)
    test_loss, test_accuracy = model.evaluate(load("test"), verbose=0)

    items = []
    matrix = np.zeros((2, 2), dtype=int)
    for image_path in sorted((DATA / "test").glob("*/*.jpg")):
        with Image.open(image_path) as image:
            result = predict_fruit(image, model)
        actual = image_path.parent.name.capitalize()
        matrix[0 if actual == "Apple" else 1, 0 if result["label"] == "Apple" else 1] += 1
        items.append({"path": str(image_path), "expected_label": actual,
                      "predicted_label": result["label"], "confidence": result["confidence"]})

    EXAMPLES.mkdir(parents=True, exist_ok=True)
    selected = []
    for fruit in ["Apple", "Orange"]:
        images = [item for item in items if item["expected_label"] == fruit]
        selected.extend([images[0], images[-1]])
    four = []
    for i, item in enumerate(selected, 1):
        target = EXAMPLES / ("test_%d_%s.jpg" % (i, item["expected_label"].lower()))
        shutil.copy2(item["path"], target)
        four.append({**item, "image_path": str(target.relative_to(ROOT))})

    history = {key: [float(value) for value in values]
               for key, values in training.history.items()}
    save_figures(history, matrix, four)
    result = {
        "trained": True, "dataset_source": manifest["source"],
        "dataset_license": manifest["license"],
        "train_images": manifest["counts"]["train"],
        "validation_images": manifest["counts"]["val"],
        "test_images": len(items),
        "requested_epochs": args.epochs, "completed_epochs": len(history["loss"]),
        "image_size": list(IMAGE_SIZE), "batch_size": 16,
        "optimizer": "Adam", "learning_rate": .001,
        "conv_filters": [24, 48, 96], "dense_units": 64,
        "model_parameters": model.count_params(),
        "test_accuracy": float(test_accuracy), "test_loss": float(test_loss),
        "test_confusion_matrix": matrix.tolist(),
        "history": history, "four_test_images": four
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Trained model:", MODEL)
    print("Measured test accuracy: %.2f%%" % (100 * test_accuracy))
    print("Next: streamlit run app.py")

if __name__ == "__main__":
    main()
