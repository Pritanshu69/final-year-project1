import cv2
import numpy as np
from tensorflow.keras.models import load_model
from utils import preprocess_frame

# Load your trained model
model = load_model("ASLmodelF.h5")

# Labels for A-Z
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed = preprocess_frame(frame)

    if processed is not None:
        preds = model.predict(processed, verbose=0)
        pred_index = np.argmax(preds)

        # ✅ Safety check: Ensure index is within range
        if pred_index < len(labels):
            pred_label = labels[pred_index]
            cv2.putText(frame, f"Predicted: {pred_label}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Prediction Error", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "No Hand Detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("ASL Recognition", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
