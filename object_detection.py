# Idea:
# 1. Estimate the background with median filtering
# 2. Remove the background from the image
# 3. Apply blurring and thresholding techniques
# 4. Detect the contours
import numpy as np
import cv2
from tracker import *
import matplotlib.pyplot as plt

np.random.seed(42)


#video_writer = cv2.VideoWriter('obj_detect.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30, (640, 480))
video_stream = cv2.VideoCapture(r'C:\Users\GHANEM\Desktop\OpenCV\data\highway_short.mp4')

# Method1.
#object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
# Method2. we get some random frames for the background

#Randomly select 30 frames
frameIds = video_stream.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=30)

# Store selected frames in an array
frames = []
for fid in frameIds:
    video_stream.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = video_stream.read()
    roi = frame[125:360, 75:500]
    frames.append(roi)

video_stream.release()
video_stream = cv2.VideoCapture(r'C:\Users\GHANEM\Desktop\OpenCV\data\highway_short.mp4')
# Calculate the median along the time axis
median_frame = np.median(frames, axis=0).astype(dtype=np.uint8)
gray_median_frame = cv2.cvtColor(median_frame, cv2.COLOR_BGR2GRAY)
#plt.imshow(cv2.cvtColor(median_frame, cv2.COLOR_BGR2RGB))
#plt.show()

# Create tracker object
tracker = EuclideanDistTracker()

# CAP_PROP_FRAME COUNT gets the nr of frames in the video file
total_frames = video_stream.get(cv2.CAP_PROP_FRAME_COUNT)  # * np.random.uniform(size=40)
car_count = 0
while True:
    ret, frame = video_stream.read()
    #height, width, _ = frame.shape
    # print(height, width) #w 75- 500, h 150-360 to find the roi with eyes
    roi = frame[125:360, 75:500]
    #mask = object_detector.apply(roi)
    #contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #for cont in contours:
        #area = cv2.contourArea(cont)
        #if area > 200:
        #    cv2.drawContours(frame, cont, -1, (0, 255, 0), 2)
    gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    abs_diff_frame = cv2.absdiff(gray_frame, gray_median_frame)
    blurred = cv2.GaussianBlur(abs_diff_frame, (11, 11), 0)
    _, threshholded_frame = cv2.threshold(blurred, 0, 255, cv2.RETR_EXTERNAL + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshholded_frame.copy(), cv2.RETR_EXTERNAL, cv2 .CHAIN_APPROX_SIMPLE)
    cv2.line(frame, (140, 200), (400, 200), (255, 0, 0), 2)
    cv2.line(frame, (140, 202), (400, 210), (0, 255, 0), 2)
    cv2.line(frame, (140, 197), (400, 190), (0, 255, 0), 2)
    detections = []
    for cont in contours:
        area = cv2.contourArea(cont)
        x, y, w, h = cv2.boundingRect(cont)
        if area > 200:  # Disregard items with too small bbox
            cv2.rectangle(roi, (x, y), (x + w, y+ h), (0, 255, 0), 2)
            xMid = int((x + (x + w)) / 2)
            yMid = int((y + (y + h)) / 2)

            detections.append([x, y, w, h])
        cv2.circle(roi, (xMid, yMid), 5, (0, 0, 255), 1)
        if 197 < yMid < 202:
            car_count += 1
    # Obj Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x - 20, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.putText(frame, 'car count: {}'.format(car_count), (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow('Frame', frame)
    #cv2.imshow('Frame', roi)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
#    video_writer.write((cv2.resize(frame, (640, 360))))

#video_stream.release()
#video_writer.release()

