import sys
import cv2 
sys.path.append('.')
from conv_recognition.head_pose import PoseEstimator

path = 'data/scene1.mp4'
cap = cv2.VideoCapture(path)

estimator = PoseEstimator()

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = estimator.get_poses(frame)
    
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
