import cv2
import time


def run_face_detection(frame):
    haar_cascades_path = \
        r"C:\Users\GHANEM\anaconda3\envs\facebook-detectron\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
    eye_cascade_path = \
        r"C:\Users\GHANEM\anaconda3\envs\facebook-detectron\Lib\site-packages\cv2\data\haarcascade_eye.xml"

    face_cascade = cv2.CascadeClassifier(haar_cascades_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        print("Face is detected")
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            print("Eyes are detected")
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    time.sleep(0.5)

    return frame
