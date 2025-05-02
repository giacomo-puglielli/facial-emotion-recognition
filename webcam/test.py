import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array

# 1) Carica architettura e pesi 
with open("webcam/models/model.json", "r") as f:
    model = model_from_json(f.read())
model.load_weights("webcam/models/model_3.h5")

# 2) Inizializza Haar-cascade
face_cascade = cv2.CascadeClassifier("webcam/models/haarcascade_frontalface_default.xml")

# 3) Apri la webcam provando diversi backend
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
if not cap.isOpened():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Impossibile aprire la webcam – controlla indice o permessi")

# 4) Loop di lettura e inferenza
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi = gray[y : y + h, x : x + w]
        roi = cv2.resize(roi, (48, 48))
        arr = img_to_array(roi)
        arr = np.expand_dims(arr, axis=0) / 255.0

        preds = model.predict(arr)
        i = np.argmax(preds[0])
        label = ['angry','disgust','fear','happy','sad','surprise','neutral'][i]
        conf  = float(np.max(preds) * 100)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"{label} {conf:.1f}%", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
