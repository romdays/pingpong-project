import numpy as np
import cv2
import re
import csv

from camera import Camera
from detection import detection
from detection import vsplit_ds_frame
from corner import mark_points_of_corners
from selection import extract_points_similarly_movements
from selection import narrow_down_by_existence_area
from selection import calc_closest_point_nearby_prev_points
from plot import PingpongPlot

from settings import Settings


def calc_corresponding_points(point_list_A, point_list_B, camera_A, camera_B):
    pairs = []
    if len(point_list_A) * len(point_list_B) == 0: return pairs

    vecs_A = camera_A.vecs_PoV2objects(point_list_A)
    vecs_B = camera_B.vecs_PoV2objects(point_list_B)

    vec_AB = camera_B.camera_position-camera_A.camera_position

    # calc distance between 2vecs
    for idx_a, vec_A in enumerate(vecs_A):
        for idx_b, vec_B in enumerate(vecs_B):
            cross = np.cross(vec_A.T,vec_B.T)
            cross = cross/np.linalg.norm(cross)

            if np.abs(np.dot(cross, vec_AB)[0,0]) < Settings.get('MAX_DISTANCE'):
                pairs.append((point_list_A[idx_a], point_list_B[idx_b]))

    return pairs


def triangulate(point_A, point_B, camera_A, camera_B):
    world_point_projected = cv2.triangulatePoints(camera_A.projection_matrix, camera_B.projection_matrix, point_A, point_B)
    world_point_projected = world_point_projected[:3] / world_point_projected[3]

    return np.array(world_point_projected)

def main():
    image_shape = (640, 480)
    Settings(image_shape)
    cameras = []
    images  = [[],[]]
    detected = [[],[]]
    points  = [[]]
    points_seq = []

    check_frame_length = 9
    prev_points_holder_2d = [[[] for i in range(check_frame_length)] for i in range(2)]
    prev_points_holder_3d = [[] for i in range(check_frame_length)]
    points_holder_2d = [[],[]]

    # load 2 videos ------------------------------------------
    cap = cv2.VideoCapture('./data/videos/ds/11.mov')
    FRAME_INTERVAL = 4#int(-(-cap.get(cv2.CAP_PROP_FPS)//60))
    
    for i in range(int(60*9.5)):
        ret, frame = cap.read()

    # detect feature point from 2 views
    for i in range(FRAME_INTERVAL*2):
        ret, frame = cap.read()
        if frame is None:
            exit()
        for i, image in enumerate(vsplit_ds_frame(frame, image_shape)):
            images[i].append(image)

    # calib by marking 2 tables corners
    corners = np.load('./data/npz/corners.npz')
    points_of_corners = [corners['first'], corners['second']]
    for i in range(2):
        # tmp_img = np.copy(images[i][0])
        # for p in points_of_corners[i]:
        #     cv2.circle(tmp_img, (int(p[0]), int(p[1])), 3, (0, 0, 255), 3)
        # print('Use these points as corners ? [y/N]')
        # cv2.imshow('check the corner', tmp_img)
        # while True:
        #     key = cv2.waitKey() & 0xFF
        #     if re.match(r'n', chr(key), re.IGNORECASE):
        #         points_of_corners[i] = mark_points_of_corners(images[i][0])
        #         np.savez('./data/npz/corners', first = points_of_corners[0], second = points_of_corners[1])
        #         break
        #     elif re.match(r'y', chr(key), re.IGNORECASE):
        #         break
        # cv2.destroyAllWindows()

        # get camera params
        cameras.append(Camera(points_of_corners[i]))

    outputter = PingpongPlot(cameras)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            break
        
        for i, image in enumerate(vsplit_ds_frame(frame, image_shape)):
            images[i].append(image)
            detected[i].append(detection(images[i][0::FRAME_INTERVAL], name=str(i)))
            images[i].pop(0)
        
        for i in range(2):
            points_holder_2d[i], prev_points_holder_2d[i] = extract_points_similarly_movements(detected[i][-check_frame_length:], prev_points_holder_2d[i])

        # calc correspond objs
        pairs = calc_corresponding_points(points_holder_2d[0], points_holder_2d[1], cameras[0], cameras[1])
        # pairs = calc_corresponding_points(detected[0][-1], detected[1][-1], cameras[0], cameras[1])

        # calc true pair point --------------------------------------------------------------------------------

        # get 3d point 
        balls = []
        for pair in pairs:
            balls.append(triangulate(pair[0], pair[1], cameras[0], cameras[1]))

        balls = narrow_down_by_existence_area(balls)
        points_seq.append(balls)


        points_holder_3d, prev_points_holder_3d = extract_points_similarly_movements(points_seq[-check_frame_length:], prev_points_holder_3d)
        points.append(calc_closest_point_nearby_prev_points(points[-5:], points_holder_3d))
        outputter.plot(points[-1])
        # outputter.plot(balls)


    # np.savez('./data/npz/points', points = points, points_seq = points_seq)
    cap.release()


if __name__ == '__main__':
    main()