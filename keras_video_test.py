# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import cv2


# 얼굴 탐지
conda_path = 'C:\\Users\\BIT-USER\\Anaconda3\\Library\\etc\\haarcascades\\'
face_cascade = cv2.CascadeClassifier(conda_path + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture('C:/Users/BIT-USER/Desktop/python_workplace/moms.mp4')
model = load_model('face_ex.model')
scaling_factor = 0.5

while True:
    ret, frame = cap.read()

    if ret:
        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in face_rects:
            img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            img = cv2.resize(img2, (28, 28))
            img = img.astype("float") / 255.0
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)

            (not_face, face) = model.predict(img)[0]

            label = "face" if face > not_face else "Not face"
            proba = face if face > not_face else not_face
            label = "{}: {:.2f}%".format(label, proba * 100)
            output = imutils.resize(frame, width=400)
            cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Face Detector', frame)

        if cv2.waitKey(1) == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
