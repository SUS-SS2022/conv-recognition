import os
import cv2
from tqdm import tqdm

scene_names = [name for name in os.listdir('data/scenes') if name.endswith('.mp4')]
scene_names.sort()
for scene_path in scene_names:
    scene_name = scene_path[:-4]
    print(f'creating image folder {scene_name}')
    root = 'data/datasets'
    if not os.path.exists(f'{root}/{scene_name}'):
        os.mkdir(f'{root}/{scene_name}')

    cap = cv2.VideoCapture(f'data/scenes/{scene_path}')

    counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f'{root}/{scene_name}/{counter:05d}.jpg', frame)
        counter += 1
    cap.release()