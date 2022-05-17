import sys
import cv2 
sys.path.append('.')
from conv_recognition.pose_estimator import PoseEstimator
from conv_recognition.laeo import get_colors
import os

scene_names = [name for name in os.listdir('data/scenes') if name.endswith('.mp4')]
scene_names.sort()

estimator = PoseEstimator()

for scene_path in scene_names:
    scene_name = scene_path[:-4]
    print(f'creating video for {scene_name}')
    root = 'data/vis'

    cap = cv2.VideoCapture(f'data/scenes/{scene_path}')
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(f'{root}/{scene_path}', fourcc, 25, (1280, 720))

    counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        head_poses = estimator.get_poses(frame)
        colors = get_colors(head_poses)
        estimator.draw_bbox(frame, head_poses, inplace=True, colors=colors)
        estimator.draw_viewing_direction(frame, head_poses, inplace=True, colors=colors)

        # cv2.imwrite(f'{root}/{scene_name}/{counter:05d}.jpg', frame)
        writer.write(frame)
        counter += 1
    cap.release()




