import sys
import cv2 
sys.path.append('.')
from conv_recognition.pose_estimator import PoseEstimator
from conv_recognition.laeo import get_colors

path = 'data/scenes/scene2.mp4'
cap = cv2.VideoCapture(path)

estimator = PoseEstimator()

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    head_poses = estimator.get_poses(frame)
    colors = get_colors(head_poses)
    estimator.draw_bbox(frame, head_poses, inplace=True, colors=colors)
    estimator.draw_viewing_direction(frame, head_poses, inplace=True, colors=colors)
    
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()


