import numpy as np
import cv2
from camera import Camera
from detection import detection
from detection import vsplit_ds_frame

from settings import (
    CALIB_PARAM,
    MAX_DISTANCE,
    FRAME_INTERVAL, 
)

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

            if np.abs(np.dot(cross, vec_AB)[0,0]) < MAX_DISTANCE:
                pairs.append((point_list_A[idx_a], point_list_B[idx_b]))
    
    return pairs

def calc_closest_point_in_pairs(pairs):
    if len(pairs)==0: return []
    distance = [pair[0]**2 + pair[1]**2 for pair in pairs]
    return pairs[distance.index(max(distance))]

def triangulate(point_A, point_B, camera_A, camera_B):
    world_point_projected = cv2.triangulatePoints(camera_A.projection_matrix, camera_B.projection_matrix, point_A, point_B)
    world_point_projected = world_point_projected[:3] / world_point_projected[3]
    
    return world_point_projected

def main():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from settings import (
        TABLE_POINTS
    )
    fig = plt.figure()
    ax = Axes3D(fig)
    x = [TABLE_POINTS[0,0], TABLE_POINTS[1,0], TABLE_POINTS[2,0], TABLE_POINTS[3,0]]
    y = [TABLE_POINTS[0,1], TABLE_POINTS[1,1], TABLE_POINTS[2,1], TABLE_POINTS[3,1]]
    z = [TABLE_POINTS[0,2], TABLE_POINTS[1,2], TABLE_POINTS[2,2], TABLE_POINTS[3,2]]
    max_range = np.array([max(x)-min(x), max(y)-min(y), max(z)-min(z)]).max() * 1.5


    cameras = []
    images  = [[],[]]
    points  = []

    # load 2 videos ------------------------------------------
    video = cv2.VideoCapture('./data/videos/ds/13.mov')

    # detect feature point from 2 views
    for i in range(FRAME_INTERVAL*2):
        ret, frame = video.read()
        top, btm = vsplit_ds_frame(frame, (640, 480))#########
        images[0].append(top)
        images[1].append(btm)
        if frame is None:
            exit()

    # calib by marking 2 tables corners ------------------------------
    points_of_corners_from_cameraA = [(83,204), (65,306), (520,304), (522,203)]
    points_of_corners_from_cameraB = [(90,159), (96,262), (550,262), (529,159)]

    # get 2 camera params
    cameras.append(Camera(points_of_corners_from_cameraA))
    cameras.append(Camera(points_of_corners_from_cameraB))

    
    while(video.isOpened()):
        point_lists  = []
        top, btm = vsplit_ds_frame(frame, (640, 480))############
        ret, frame = video.read()
        images[0].append(top)
        images[1].append(btm)
        if frame is None:
            break

            
        point_lists.append(detection(images[0][0::FRAME_INTERVAL]))
        point_lists.append(detection(images[1][0::FRAME_INTERVAL]))
        images[0].pop(0)
        images[1].pop(0)

        # calc correspond objs
        pairs = calc_corresponding_points(point_lists[0], point_lists[1], cameras[0], cameras[1])

        # calc true pair point --------------------------------------------------------------------------------

        cand = []
        for pair in pairs:
            cand.append(triangulate(pair[0], pair[1], cameras[0], cameras[1]))

        if len(cand)>0:
            # get 3d point 
            # points.append(triangulate(pair[0], pair[1], cameras[0], cameras[1]))
            points.append(calc_closest_point_in_pairs(cand))

        else:
            points.append(np.array([[0.],[0.],[0.]]))
            # if no detected point, do interpolate
            cv2.waitKey(1)
            continue


        ax.cla()
        ax.set_xlim(-max_range/2, max_range/2)
        ax.set_ylim(-max_range/2, max_range/2)
        ax.set_zlim(0, max_range)
        x.append(points[-1][0,0])
        y.append(points[-1][1,0])
        z.append(points[-1][2,0])
        ax.scatter(x,y,z)

        plt.pause(.01)
        x.pop()
        y.pop()
        z.pop()


    video.release()


if __name__ == '__main__':
    main()