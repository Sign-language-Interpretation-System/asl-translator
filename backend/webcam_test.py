import cv2
import mediapipe as mp
import numpy as np
from collections import deque, Counter

history = deque(maxlen=10)
from utils.predict import predict

# =========================
# MediaPipe Setup
# =========================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =========================
# Webcam Setup
# =========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

# =========================
# Main Loop
# =========================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb_frame)

    prediction_text = "No Hand Detected"

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            h, w, _ = frame.shape

            x_coords = []
            y_coords = []

            x_coords = []
            y_coords = []

            for lm in hand_landmarks.landmark:

                x = int(lm.x * w)
                y = int(lm.y * h)

                x_coords.append(x)
                y_coords.append(y)
            padding = 30
        

            x_min = min(x_coords)
            y_min = min(y_coords)

            x_max = max(x_coords)
            y_max = max(y_coords)

            box_size = max(
                x_max - x_min,
                y_max - y_min
            )

            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2

            x_min = max(center_x - box_size // 2 - padding, 0)
            y_min = max(center_y - box_size // 2 - padding, 0)

            x_max = min(center_x + box_size // 2 + padding, w)
            y_max = min(center_y + box_size // 2 + padding, h)

            # Draw box around hand
            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                (0, 255, 0),
                2
            )

            # Crop hand
            hand_crop = frame[
                y_min:y_max,
                x_min:x_max
            ]

            hand_crop = cv2.cvtColor(
                hand_crop,
                cv2.COLOR_BGR2RGB
            )



            if hand_crop.size > 0:

                preview = cv2.resize(hand_crop, (256, 256))
                cv2.imshow("Hand Crop", preview)

                hand_crop = cv2.resize(
                    hand_crop,
                    (64, 64)
                )

                # Normalize
                hand_crop = hand_crop.astype(
                    np.float32
                ) / 255.0

                # Add batch dimension
                hand_crop = np.expand_dims(
                    hand_crop,
                    axis=0
                )


                try:

                    letter, confidence, top5 = predict(hand_crop)
                    if confidence > 85:
                        history.append(letter)

                    if len(history) > 0:
                        stable_letter = Counter(history).most_common(1)[0][0]
                    else:
                        stable_letter = letter

                    print("Top 5 Predictions:")
                    print(top5)
                    print("-" * 50)

                    prediction_text = (
                        f"{stable_letter} "
                    )
                    history.append(letter)
                    stable_letter = Counter(history).most_common(1)[0][0]

                except Exception as e:

                    prediction_text = (
                        f"Prediction Error"
                    )

                    print(e)

            break


    # Display prediction
    cv2.putText(
        frame,
        prediction_text,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        3,
        (0, 0, 0),
        5
    )

    cv2.imshow(
        "ASL Finger Spelling Recognition",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# =========================
# Cleanup
# =========================

cap.release()
cv2.destroyAllWindows()