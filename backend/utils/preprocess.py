import numpy as np
from PIL import Image
import io

IMG_SIZE = (64, 64)

def preprocess_image(image_bytes):
    """
    Takes raw image bytes, returns a normalized numpy array ready for inference.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # shape: (1, 64, 64, 3)