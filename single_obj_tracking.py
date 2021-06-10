import cv2

# Trackers to try
TrackerDict = {'csrt': cv2.TrackerCSRT_create,
               'kcf': cv2.TrackerKCF_create,
               'boosting': cv2.TrackerBoosting_create,
               'mil': cv2.TrackerMIL_create,
               'tld': cv2.TrackerTLD_create,
               'mosse': cv2.TrackerMOSSE_create}
# init the tracker and import the video
tracker = TrackerDict['mil']()
video = cv2.VideoCapture(r'C:\Users\GHANEM\Desktop\OpenCV\data\highway_short.mp4')

# view first frame -> choose roi -> init the tracker from that frame with that bbox
ret, frame = video.read()
cv2.imshow('Frame', frame)
BBox = cv2.selectROI('ROI', frame, True)
tracker.init(frame, BBox)

# now to keep the bbox on the obj wherever it goes
while True:
    ret, frame = video.read()
    if not ret:
        break
    success, box = tracker.update(frame)
    if success:
        (x, y, w, h) = [int(a) for a in box]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)  # (x,y)(top left) , (x+w, y+h)(bottom right)
    cv2.imshow('Frame', frame)
    # to close
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
