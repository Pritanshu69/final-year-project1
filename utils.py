import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

def preprocess_frame(frame):
    """Extract 21 hand landmarks (x, y) = 42 features for model input"""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        features = []
        for lm in hand_landmarks.landmark:
            features.extend([lm.x, lm.y])  # take x, y → 42 values
        return np.array(features).reshape(1, -1)  # shape (1, 42)

    return None  # if no hand detected
