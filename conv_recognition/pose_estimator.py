from torchvision import transforms
from conv_recognition.model import SixDRepNet
from face_detection import RetinaFace
import torch
from PIL import Image
import conv_recognition.utils as utils
import numpy as np
from copy import copy

class PoseEstimator:

    def __init__(self, gpu=0, snapshot_path='data/models/6DRepNet_300W_LP_AFLW2000.pth'):
        self.transformations = transforms.Compose([transforms.Resize(224),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.detector = RetinaFace(gpu_id=gpu)
        self.model = SixDRepNet(backbone_name='RepVGG-B1g2',
                                backbone_file='',
                                deploy=True,
                                pretrained=False)
        saved_state_dict = torch.load(snapshot_path, map_location='cpu')
        if 'model_state_dict' in saved_state_dict:
            self.model.load_state_dict(saved_state_dict['model_state_dict'])
        else:
            self.model.load_state_dict(saved_state_dict)
        self.model.cuda(gpu)
        self.model.eval()
        self.gpu = gpu

    def get_poses(self, frame):
        faces = self.detector(frame)
        head_poses = []
        with torch.no_grad():
            for box, landmarks, score in faces:
                if score < 0.95:
                    continue
                x_min = int(box[0])
                y_min = int(box[1])
                x_max = int(box[2])
                y_max = int(box[3])
                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)

                x_min = max(0, x_min-int(0.2*bbox_height))
                y_min = max(0, y_min-int(0.2*bbox_width))
                x_max = x_max+int(0.2*bbox_height)
                y_max = y_max+int(0.2*bbox_width)

                img = frame[y_min:y_max, x_min:x_max]
                img = Image.fromarray(img)
                img = img.convert('RGB')
                img = self.transformations(img)

                img = torch.Tensor(img[None, :]).cuda(self.gpu)

                pred = self.model(img)
                euler = utils.compute_euler_angles_from_rotation_matrices(
                    pred)*180/np.pi

                pitch = euler[:, 0].cpu()
                yaw = euler[:, 1].cpu()
                roll = euler[:, 2].cpu()

                center_x = x_min + int(.5*(x_max-x_min))
                center_y = y_min + int(.5*(y_max-y_min))

                head_poses.append((center_x, center_y, bbox_width, yaw, pitch, roll))
        return head_poses

    def draw_head_poses(self, frame, head_poses):
        frame = copy(frame)
        for center_x, center_y, size, yaw, pitch, roll in head_poses:
            utils.plot_pose_cube(
                    frame,  yaw, pitch, roll,
                    center_x,
                    center_y,
                    size=size)
        return frame

    def get_looking_direction(self, frame):
        return
