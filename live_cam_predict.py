import cv2
import numpy as np
from keras.models import model_from_json

print("Starting program...")

# =========================
# Emotion labels (FER2013)
# =========================
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# =========================
# Load model
# =========================
print("Loading model...")
with open('top_models/fer.json', 'r') as f:
    model = model_from_json(f.read())

model.load_weights('top_models/fer.h5')
print("Model loaded")

# =========================
# Load Haar Cascade
# =========================
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("❌ Haar cascade not loaded")
    exit()

# =========================
# Open Camera
# =========================
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ Camera not opened")
    exit()

print("✅ Camera opened")

# =========================
# Main Loop
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame not received")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))

        roi_gray = roi_gray.astype("float") / 255.0
        roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))

        preds = model.predict(roi_gray, verbose=0)
        emotion = emotions[np.argmax(preds)]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            emotion,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2
        )

    cv2.imshow("Live Facial Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# Cleanup
# =========================
cap.release()
cv2.destroyAllWindows()
print("Program closed")
