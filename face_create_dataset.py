import cv2
import os

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
dataset_path = "dataset/"

if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)

person_id = 1  # id untuk orang yang akan terdeteksi
count = 0      # menghitung jumlah id gambar

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1
        cv2.imwrite(dataset_path + "Person-" + str(person_id) + "-" + str(count) + ".jpg", gray[y:y + h, x:x + w])

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord('q'):
        break
    elif count == 30:  # berhenti setelah 30 foto diambil
        break

cap.release()
cv2.destroyAllWindows()