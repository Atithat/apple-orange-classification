"""Shared model preprocessing for website and held-out evaluation."""
import numpy as np
from PIL import ImageOps
from model import IMAGE_SIZE

def predict_fruit(image, model):
    image = ImageOps.exif_transpose(image).convert("RGB").resize(IMAGE_SIZE)
    batch = np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)
    orange_probability = float(np.asarray(model.predict(batch, verbose=0)).reshape(-1)[0])
    orange_probability = max(0.0, min(1.0, orange_probability))
    scores = {"Apple": 1.0 - orange_probability, "Orange": orange_probability}
    label = max(scores, key=scores.get)
    return {"label": label, "confidence": scores[label], "scores": scores}
