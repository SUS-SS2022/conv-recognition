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
    parser.add_argument('prediction',
                        help='path for prediction')
    parser.add_argument('dst',
                        help='destination folder for results')
    
    args = parser.parse_args()

    f = open(args.dst, 'w+')
    with open(args.prediction, 'r') as file:
        predictions = [line.split(' ') for line in file.readlines()]
        predictions = [f'{name} {label}' for name, label in predictions]

    estimator = PoseEstimator()

    cap = cv2.VideoCapture(args.video)

    fontScale = 1
    color = (255, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    org = (0, 50)

    counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        prediction = predictions[counter]
        frame = cv2.putText(frame, prediction, org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
        if not ret:
            break
        cv2.imshow('frame', frame)
        key = cv2.waitKey(0)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('y'):
            label = 1
        else:
            label = 0
        f.write(f'{counter:05d}.jpg {label}\n')
        f.flush()
        counter+=1
    f.close()

        


    
