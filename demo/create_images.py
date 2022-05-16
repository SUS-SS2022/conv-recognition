import os
import cv2

for scene_path in os.listdir('data/scenes'):
    scene_name = scene_path[:-4]
    if not os.path.exists(f'data/images/{scene_name}'):
        os.mkdir(f'data/images/{scene_name}')
    cap = cv2.VideoCapture(f'data/scenes/{scene_path}')

    counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f'data/images/{scene_name}/{counter}.jpg', frame)
        counter += 1
    cap.release()