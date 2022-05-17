import sys
sys.path.append('.')

import argparse
import os
import cv2
import cv2 
from conv_recognition.pose_estimator import PoseEstimator
from conv_recognition.laeo import find_laeo, find_intersections
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Label images with laeo detector')
    parser.add_argument('video',
                        help='source video/image folder')
    parser.add_argument('prediction', required=False,
                        help='path for prediction')
    parser.add_argument('dst',
                        help='destination folder for results')
    parser.add_argument('--video', required=False,
                        help='if set, create video at given path')
    
    args = parser.parse_args()

    if os.path.exists(args.dst):
        os.remove(args.dst)

    f = open(args.dst, 'w+')
    file_names = os.listdir(args.source)
    file_names.sort()

    estimator = PoseEstimator()

    counter = 0
    for file_name in tqdm(file_names):
        frame = cv2.imread(f'{args.source}/{file_name}')
        head_poses = estimator.get_poses(frame)
        intersections = find_intersections(head_poses)
        laeo = find_laeo(intersections)
        label = 1 if laeo else 0
        f.write(f'{counter:05d}.jpg {label}\n')
        f.flush()
        counter+=1
    f.close()