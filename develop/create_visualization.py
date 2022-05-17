import os
import cv2
from tqdm import tqdm

video_directory = 'data/vis'
image_directory = 'data/vis_images'
scene_names = [name for name in os.listdir(video_directory) if name.endswith('.mp4')]
scene_names.sort()
for scene_path in scene_names:
    scene_name = scene_path[:-4]
    print(f'creating image folder {scene_name}')
    if not os.path.exists(f'{image_directory}/{scene_name}'):
        os.mkdir(f'{image_directory}/{scene_name}')

    cap = cv2.VideoCapture(f'{video_directory}/{scene_path}')

    counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f'{image_directory}/{scene_name}/{counter:05d}.jpg', frame)
        counter += 1
    cap.release()