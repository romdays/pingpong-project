import numpy as np
import cv2
from camera import Camera
from detection import detection

from settings import (
    CALIB_PARAM,
    MAX_DISTANCE,
    FRAME_INTERVAL, 
)

def calc_corresponding_points(point_list_A, point_list_B, camera_A, camera_B):
    pairs = []
    if len(point_list_A)*len(point_list_B)==0: return pairs
    
    vecs_A = camera_A.vecs_PoV2objects(point_list_A)
    vecs_B = camera_B.vecs_PoV2objects(point_list_B)

    for idx_a, vec_A in enumerate(vecs_A):
        for idx_b, vec_B in enumerate(vecs_B):
            cross = np.cross(vec_A.T,vec_B.T)
            cross = cross/np.linalg.norm(cross)

            vec_AB = camera_B.camera_position-camera_A.camera_position

            if np.abs(np.dot(cross, vec_AB)[0,0]) < MAX_DISTANCE:
                pairs.append((point_list_A[idx_a], point_list_B[idx_b]))
    
    return pairs

def triangulate(point_A, point_B, camera_A, camera_B):
    world_point_projected = cv2.triangulatePoints(camera_A.projection_matrix, camera_B.projection_matrix, point_A, point_B)
    world_point_projected = world_point_projected[:3] / world_point_projected[3]
    
    return world_point_projected

def main():
    videos  = []
    cameras = []
    images  = [[],[]]
    points  = []

    # load 2 videos ------------------------------------------
    videos.append(cv2.VideoCapture('./data/videos/PoVA.mp4'))
    videos.append(cv2.VideoCapture('./data/videos/PoVB.mp4'))

    # detect feature point from 2 views
    for video, seq in zip(videos, images):
        for i in range(FRAME_INTERVAL*2):
            ret, frame = video.read()
            seq.append(frame)
            if frame is None:
                exit()

    # calib by marking 2 tables corners ------------------------------
    points_of_corners_from_cameraA = [(10,10), (20,20), (30,30), (40,40)]
    points_of_corners_from_cameraB = [(10,10), (20,20), (30,30), (40,40)]

    # get 2 camera params
    cameras.append(Camera(points_of_corners_from_cameraA))
    cameras.append(Camera(points_of_corners_from_cameraB))

    
    flag = False
    while(videos[0].isOpened() and videos[1].isOpened()):
        point_lists  = [[],[]]
        for video, seq, point_list in zip(videos, images, point_lists):
            ret, frame = video.read()
            seq.append(frame)
            if frame is None:
                flag = True
                break
            
            point_list.append(detection(seq[0::FRAME_INTERVAL]))
            seq.pop(0)

        if flag:# or cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # calc correspond objs
        pairs = calc_corresponding_points(point_lists[0], point_lists[1], cameras[0], cameras[1])

        # calc true pair point --------------------------------------------------------------------------------

        # get 3d point 
        points.append = triangulate(pairs[0][0], pairs[0][1], cameras[0], cameras[1])

        # if no detected point, do interpolate


    for video in videos:
        video.release()


if __name__ == '__main__':
    main()