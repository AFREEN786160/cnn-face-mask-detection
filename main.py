import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("face_mask_model.h5")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized = cv2.resize(frame, (128, 128))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 128, 128, 3))

    prediction = model.predict(reshaped)

    if prediction[0][0] > 0.5:
        label = "Real Face"
    else:
        label = "Disguise Mask"

    cv2.putText(frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
