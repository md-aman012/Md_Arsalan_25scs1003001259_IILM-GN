import argparse
import cv2
import numpy as np
from keras.models import model_from_json
from tensorflow.keras.utils import img_to_array

# ===============================
# Emotion labels (FER2013)
# ===============================
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ===============================
# Parse video source argument
# ===============================
ap = argparse.ArgumentParser()
ap.add_argument('video', help='camera index (0, 1, 2...) or video file path')
args = vars(ap.parse_args())

# ===============================
# Load model from JSON
# ===============================
json_file = open('top_models\\fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
model.load_weights('top_models\\fer.h5')

# ===============================
# Load Haar Cascade
# ===============================
face_haar_cascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml'
)

# ===============================
# Open Camera
# ===============================
cap = cv2.VideoCapture(int(args['video']))
print("Camera opened:", cap.isOpened())

if not cap.isOpened():
    print("❌ Camera not accessible")
    exit()

print("Entering camera loop")

# ===============================
# Main Loop
# ===============================
while True:
    ret, frame = cap.read()

    if frame is None:
        print("Empty frame, skipping...")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_haar_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))

        img_pixels = img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255.0

        preds = model.predict(img_pixels, verbose=0)
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

    cv2.imshow("Facial Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===============================
# Cleanup
# ===============================
cap.release()
cv2.destroyAllWindows()
