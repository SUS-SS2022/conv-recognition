import sys
sys.path.append('.')

import argparse
import os
import cv2
import cv2 
from conv_recognition.pose_estimator import PoseEstimator
from conv_recognition.laeo import find_laeo, find_intersections, get_colors
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Label images with laeo detector')
    parser.add_argument('source',
                        help='source video')
    parser.add_argument('--prediction',
                        help='destination folder for prediction results')
    parser.add_argument('--vis', required=False, action='store_true',
                        help='Flag if results should be visualized during prediction process')
    parser.add_argument('--vispath', required=False,
                        help='Path to save the visualization of the video')
    
    args = parser.parse_args()

    if args.prediction and os.path.exists(args.prediction):
        os.remove(args.prediction)

    cap = cv2.VideoCapture(args.source)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    estimator = PoseEstimator()
    
    try:
        if args.prediction:
            f = open(args.prediction, 'w+')
        if args.vispath:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(args.vispath, fourcc, fps, (width, height))
        for i in tqdm(range(frame_count)):
            ret, frame = cap.read()
            if not ret:
                print('error while reading video')
                break
            head_poses = estimator.get_poses(frame)
            intersections = find_intersections(head_poses)
            laeo = find_laeo(intersections)
            label = 1 if laeo else 0
            if args.prediction:
                f.write(f'{i} {label}\n')
                f.flush()
            if args.vis or args.vispath:
                colors = get_colors(head_poses)
                estimator.draw_bbox(frame, head_poses, inplace=True, colors=colors)
                estimator.draw_viewing_direction(frame, head_poses, True, colors)
            if args.vis:
                cv2.imshow('Prediction', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if args.vispath:
                writer.write(frame)
    except Exception as e:
        print(f'Error while reading video: {e}')
    if args.prediction:
        f.close()
    if args.vispath:
        writer.release()
    cap.release()