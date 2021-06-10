from threading import Thread
import cv2
import time
import numpy as np
import os


class VidStream:
    def __init__(self, src, width, height):
        self.width = width
        self.height = height
        self.capture = cv2.VideoCapture(src)
        self.thread = Thread(target=self.update, args=())  # a thread that will continuously read a frame from cam 1
        self.thread.daemon = True  # Daemon Thread doesn't block the main thread from exiting and continues to run in bg
        self.thread.start()

    def update(self):
        while True:
            _, self.frame = self.capture.read()
            self.frame2 = cv2.resize(self.frame, (self.width, self.height))

    def getframe(self):
        return self.frame2


models_dir = r'C:\Users\GHANEM\Desktop\OpenCV\models'
with open(os.path.join(models_dir, 'object_detection_classes_coco.txt')) as f:
    class_labels = f.read().split('\n')
colors = np.random.uniform(0, 255, size=(len(class_labels), 3))
# Load model
config_file = os.path.join(models_dir, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
frozen_model = os.path.join(models_dir, "frozen_inference_graph.pb")
model = cv2.dnn.readNet(frozen_model, config_file, 'TensorFlow')

# model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


dispW = 640
dispH = 480
cam1 = VidStream(0, dispW, dispH)
cam2 = VidStream(1, dispW, dispH)
startTime = time.time()
dtav = 0
min_confidence_score = 0.6
while True:
    try:
        myFrame1 = cam1.getframe()
        myFrame2 = cam2.getframe()
        myFrame3 = np.hstack((myFrame1, myFrame2))  # horizontally stack
        imgHeight, imgWidth, channels = myFrame3.shape
        blob = cv2.dnn.blobFromImage(myFrame3, size=(320, 320), mean=(127.5, 127.5, 127.5), scalefactor=1.0 / 127.5,
                                     swapRB=True)
        model.setInput(blob)
        output = model.forward()

        dt = time.time() - startTime
        startTime = time.time()
        dtav = 0.9 * dtav + 0.1 * dt
        fps = 1/dtav
        for detection in output[0, 0, :, :]:
            confidence = detection[2]
            if confidence > min_confidence_score:
                class_id = detection[1]
                class_name = class_labels[int(class_id) - 1]
                color = colors[int(class_id)]

                BBoxX = detection[3] * imgWidth
                BBoxY = detection[4] * imgHeight
                BBoxWidth = detection[5] * imgWidth
                BBoxHeight = detection[6] * imgHeight

                cv2.rectangle(myFrame3, (int(BBoxX), int(BBoxY)), (int(BBoxWidth), int(BBoxHeight)), color, thickness=2)
                cv2.putText(myFrame3, class_name, (int(BBoxX), int(BBoxY - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.rectangle(myFrame3, (0, 0), (140, 40), (255, 0, 0), -1)
        cv2.putText(myFrame3, str(round(fps, 1)) + 'fps', (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.imshow("SplitCam", myFrame3)
    except:
        print("no frame available")
    if cv2.waitKey(1) == ord("q"):
        cam1.capture.release()
        cam2.capture.release()
        cv2.destroyAllWindows()
        exit(1)
        break
