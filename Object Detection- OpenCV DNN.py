# DL algorithms for img classification:
# AlexNet, GoogleNet, MobileNet, VGGNet - with ImageNet dataset (1000 Classes)
# For Object Detection: (combination of classification and localization)
# SSD-MobileNet or YOLO
# https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API

import cv2
import numpy as np
import os
import time
from tracker import *
# pip install tracker

models_dir = os.path.join(os.getcwd(), 'models')
# Load class names
with open(os.path.join(models_dir, 'object_detection_classes_coco.txt')) as f:
    class_labels = f.read().split('\n')
# Get a different colors for each of the classes
colors = np.random.uniform(0, 255, size=(len(class_labels), 3))
# Load model
config_file = os.path.join(models_dir, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
frozen_model = os.path.join(models_dir, "frozen_inference_graph.pb")
# model = cv2.dnn_DetectionModel(frozen_model, config_file)
# model.setInputSize(320, 320)
# model.setInputScale(1.0/127.5)
# model.setInputMean((127.5, 127.5, 127.5))  # mobilenet takes [-1, 1]
# model.setInputSwapRB(True)
model = cv2.dnn.readNet(frozen_model, config_file, 'TensorFlow')
# Set backend and target to CUDA to use GPU
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
# Create tracker object
tracker = EuclideanDistTracker()
# Webcam
cap = cv2.VideoCapture(0)
min_confidence_score = 0.55
while cap.isOpened():
    # Read Img
    success, img = cap.read()
    imgHeight, imgWidth, channels = img.shape
    # Create blob fomr img
    blob = cv2.dnn.blobFromImage(img, size=(320, 320), mean=(127.5, 127.5, 127.5), scalefactor=1.0/127.5, swapRB=True)
    # start time to calculate FPS
    start = time.time()
    # set input to the model
    model.setInput(blob)
    # Make forward pass in model
    output = model.forward()
    # End time
    end = time.time()
    # Calculate FPS
    fps = 1 / (end-start)
    # run over each of the detections
    detections = []
    for detection in output[0, 0, :, :]:
        confidence = detection[2]
        if confidence > min_confidence_score:
            class_id = detection[1]
            class_name = class_labels[int(class_id)-1]
            color = colors[int(class_id)]

            BBoxX = detection[3] * imgWidth
            BBoxY = detection[4] * imgHeight
            BBoxWidth = detection[5] * imgWidth
            BBoxHeight = detection[6] * imgHeight
            detections.append([int(BBoxX), int(BBoxY), int(BBoxWidth), int(BBoxHeight)])

            cv2.rectangle(img, (int(BBoxX), int(BBoxY)), (int(BBoxWidth), int(BBoxHeight)), color, thickness=2)
            cv2.putText(img, class_name, (int(BBoxX), int(BBoxY - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    # Obj Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(img, str(id), (x - 20, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # Show FPS
    cv2.putText(img, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("image", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
