import os
from PIL import Image
import numpy as np

from utils.predict import predict

DATASET = "../dataset/asl_alphabet_test"

correct = 0
total = 0

for file in os.listdir(DATASET):

    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    # Example: A_test.jpg -> A
    expected = file.split("_")[0]

    path = os.path.join(DATASET, file)

    img = Image.open(path).convert("RGB")
    img = img.resize((64, 64))

    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred, confidence, _ = predict(arr)

    total += 1

    if pred == expected:
        correct += 1
        result = "✓"
    else:
        result = "✗"

    print(
        f"{file:20} "
        f"Expected={expected:8} "
        f"Predicted={pred:8} "
        f"Conf={confidence:.2f}% "
        f"{result}"
    )

print("\n===================")
print(f"Correct : {correct}")
print(f"Total   : {total}")
print(f"Accuracy: {100 * correct / total:.2f}%")