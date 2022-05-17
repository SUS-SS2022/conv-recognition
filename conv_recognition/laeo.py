import numpy as np
from math import cos, sin
from numpy.linalg import norm
from typing import List, Tuple
from conv_recognition.pose_estimator import HeadPose

def intersect(x_min, x_max, y_min, y_max, origin_x, origin_y, ray_x, ray_y):
    tmin, tmax = -np.infty, np.infty

    if ray_x != 0.0:
        tx1 = (x_min-origin_x)/ray_x
        tx2 = (x_max-origin_x)/ray_x

        tmin = max(tmin, min(tx1, tx2))
        tmax = min(tmax, max(tx1, tx2))

    if ray_y != 0.0:
        ty1 = (y_min-origin_y)/ray_y
        ty2 = (y_max-origin_y)/ray_y

        tmin = max(tmin, min(ty1, ty2))
        tmax = min(tmax, max(ty1, ty2))
    return tmax >= tmin and tmax > 0.0

def find_intersections(head_poses: HeadPose):
    intersections = []
    for i, hp in enumerate(head_poses):
        p = hp.pitch * np.pi / 180
        y = -(hp.yaw * np.pi / 180)
        ray_dir = np.array([sin(y), (-cos(y) * sin(p))])
        ray_dir /= norm(ray_dir)

        for j, hp2 in enumerate(head_poses):
            if i==j:
                continue
            x_min = hp2.center_x-hp2.bbox_width/2
            x_max = hp2.center_x+hp2.bbox_width/2
            y_min = hp2.center_y-hp2.bbox_height/2
            y_max = hp2.center_y+hp2.bbox_height/2
            if intersect(x_min, x_max, y_min, y_max, hp.center_x, hp.center_y, ray_dir[0], ray_dir[1]):
                intersections.append((i, j))
    return intersections

def find_laeo(intersections: List[Tuple[int, int]]):
    intersections_reversed = [tuple(reversed(i)) for i in intersections]
    intersections.sort()
    intersections_reversed.sort()
    laeo = []
    for i1, i2 in zip(intersections, intersections_reversed):
        if i1==i2 and tuple(reversed(i1)) not in laeo:
            laeo.append(i1)
    return laeo

def assign_colors(head_poses, laeo):
    laeo_colors = [
        (51, 153, 255),
        (51, 255, 255),
        (51, 51, 255),
        (153, 51, 255),
        (255, 51, 255),
        (255, 51, 153),
    ]
    colors = [(0, 0, 255) for _ in head_poses]
    counter = 0
    for i1, i2 in laeo:
        colors[i1] = laeo_colors[counter]
        colors[i2] = laeo_colors[counter]
        counter += 1
    return colors



def get_colors(head_poses):
    intersections = find_intersections(head_poses)
    laeo = find_laeo(intersections)
    colors = assign_colors(head_poses, laeo)
    return colors