import cv2

# Trackers to try
TrackerDict = {'csrt': cv2.TrackerCSRT_create,
               'kcf': cv2.TrackerKCF_create,
               'boosting': cv2.TrackerBoosting_create,
               'mil': cv2.TrackerMIL_create,
               'tld': cv2.TrackerTLD_create,
               'mosse': cv2.TrackerMOSSE_create}

trackers = cv2.MultiTracker_create()  # creates multiple obj container
video = cv2.VideoCapture(r'C:\Users\GHANEM\Desktop\OpenCV\data\highway_short.mp4')
ret, frame = video.read()
num_obj_to_track = 3
for obj_nr in range(num_obj_to_track):
    BBox = cv2.selectROI('Frame', frame)  # cv.SetImageROI(imag, rect)
    tracker = TrackerDict['mil']()
    trackers.add(tracker, frame, BBox)

while True:
    ret, frame = video.read()
    if not ret:
        break
    success, boxes = trackers.update(frame)
    for box in boxes:
        (x, y, w, h) = [int(a) for a in box]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)  # (x,y)(top left) , (x+w, y+h)(bottom right)
    cv2.imshow('Frame', frame)
    # to close
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
