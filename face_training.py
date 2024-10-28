import cv2
import numpy as np
import os

def checkDataset(directory="dataset/"):
    if os.path.exists(directory) and len(os.listdir(directory)) != 0:
        return True
    return False

def organizeDataset(path="dataset/"):
    imagePath = [os.path.join(path, p) for p in os.listdir(path)]
    faces = []
    ids = np.array([], dtype="int")
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    for i in imagePath:
        # Ambil ID dari nama file 'Person-ID-Count.jpg'
        file_name = os.path.basename(i)
        try:
            id = int(file_name.split("-")[1])  # Mengambil ID kedua
        except (IndexError, ValueError):
            print(f"Skipping file due to unexpected name format: {file_name}")
            continue  # Lewatkan file yang tidak sesuai format

        img = cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2GRAY)
        face = faceCascade.detectMultiScale(img)

        if len(face) == 0:
            print(f"No face detected in file: {file_name}")
            continue

        for (x, y, w, h) in face:
            faces.append(img[y:y+h, x:x+w])
            ids = np.append(ids, id)

    return faces, ids

if not checkDataset():
    print("Dataset not found")
else:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Train faces
    print("Training faces...")
    faces, ids = organizeDataset()
    
    if len(faces) == 0 or len(ids) == 0:
        print("No faces found in the dataset for training.")
    else:
        recognizer.train(faces, ids)
        print("Training finished!")

        # Save model
        recognizer.write("face-model.yml")
        print("Model saved as 'face-model.yml'")
