import json
import numpy as np
import tensorflow as tf

_model = None
_labels = None  # {label_name: index}
_idx_to_label = None

def load_model():
    global _model, _labels, _idx_to_label
    if _model is None:
        _model = tf.keras.models.load_model("model/asl_cnn_model.h5")
        with open("model/class_labels.json") as f:
            _labels = json.load(f)
        _idx_to_label = {v: k for k, v in _labels.items()}

    print("Label Mapping:")
    print(_idx_to_label)
    return _model, _idx_to_label

def predict(image_array):
    """
    image_array: numpy array of shape (1, 64, 64, 3), values in [0,1]
    Returns: (predicted_letter, confidence_percent, top5_list)
    """
    model, idx_to_label = load_model()
    preds = model.predict(image_array)[0]  # shape: (num_classes,)

    top_idx = np.argmax(preds)
    confidence = float(preds[top_idx]) * 100

    top5_indices = np.argsort(preds)[::-1][:5]
    top5 = [
        {"letter": idx_to_label[i], "confidence": round(float(preds[i]) * 100, 2)}
        for i in top5_indices
    ]

    return idx_to_label[top_idx], round(confidence, 2), top5