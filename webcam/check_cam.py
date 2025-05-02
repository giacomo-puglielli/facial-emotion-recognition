import cv2

# apri la webcam con il backend DirectShow
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Impossibile aprire la webcam")

# leggi un fotogramma di prova
ret, frame = cap.read()
print("Frame letto:", ret)
if ret:
    cv2.imshow("Test camera", frame)
    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
