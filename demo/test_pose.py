import sys
import cv2 
sys.path.append('.')
from conv_recognition.pose_estimator import PoseEstimator

path = 'data/scenes/scene2.mp4'
cap = cv2.VideoCapture(path)

estimator = PoseEstimator()

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    head_poses = estimator.get_poses(frame)
    estimator.draw_head_poses(frame, head_poses, True)
    
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
