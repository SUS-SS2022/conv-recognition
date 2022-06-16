import cv2
import numpy as np
import glob


class Dataset():

    def __init__(self, path: str):
        if path.endswith(('.mp4', '.avi')):
            self.cap = cv2.VideoCapture(path)
            self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video = True
        else:
            ext = ['png', 'jpg']
            files = []
            self.image_paths = [files.extend(glob.glob(path + '*.' + e)) for e in ext]
            self.length = len(self.image_paths)
            self.video = False
    

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> np.ndarray:
        if self.video:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    yield frame
                else:
                    break
        else:
            for i in range(len(self)):
                path = self.image_paths[i]
                frame = cv2.imread(path)
                yield frame
        