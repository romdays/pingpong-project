import numpy as np
import cv2 # Right-handed coordinate system
# import matplotlib
from camera import Camera

from settings import (
    TABLE_POINTS,
)

class PointInfo():
    def __init__(self, npoints, img):
        self.npoints = npoints
        self.points = np.empty((npoints, 2), dtype=float)
        self.pos = 0
        self.marker = []
        self.img = np.copy(img)

    def add(self, x, y):
        if self.pos < self.npoints:
            self.points[self.pos, :] = [x, y]
            self.pos += 1
            return True
        return False


def onMouse(event, x, y, flags, param):
    wname, p_info = param
    if len(p_info.marker)%2==1 and event == cv2.EVENT_MOUSEMOVE:
        tmp = np.copy(p_info.img)
        if len(p_info.marker)>1:
            cv2.line(tmp, (p_info.marker[0][0], p_info.marker[0][1]), (p_info.marker[1][0], p_info.marker[1][1]), (0, 255, 255))
        cv2.line(tmp, (p_info.marker[-1][0], p_info.marker[-1][1]), (x, y), (0, 255, 255))
        cv2.imshow(wname, tmp)

    if event == cv2.EVENT_LBUTTONDOWN:
        p_info.marker.append((x,y))
        print(x,y)

        if len(p_info.marker)==4:
            p4 = p_info.marker.pop()
            p2 = p_info.marker.pop()
            p3 = p_info.marker.pop()
            p1 = p_info.marker.pop()
            s1 = ((p4[0] - p2[0]) * (p1[1] - p2[1]) - (p4[1] - p2[1]) * (p1[0] - p2[0])) * 0.5
            s2 = ((p4[0] - p2[0]) * (p2[1] - p3[1]) - (p4[1] - p2[1]) * (p2[0] - p3[0])) * 0.5
            X = p1[0] + (p3[0] - p1[0]) * s1 / (s1 + s2)
            Y = p1[1] + (p3[1] - p1[1]) * s1 / (s1 + s2)
            if p_info.add(X,Y):
                print('[%d] ( %d, %d )' % (p_info.pos - 1, X, Y))
                cv2.circle(p_info.img, (int(X), int(Y)), 3, (0, 0, 255), 3)
                cv2.imshow(wname, p_info.img)
            else:
                print('All points have selected.  Press ESC-key.')


if __name__ == '__main__':
    img = cv2.imread('./data/images/wtt2019a.png')
    img = cv2.resize(img, (640, 480))
    npoints = 4
    p_info = PointInfo(npoints, img)
    cv2.imshow('sourceA', img)

    # # cap = cv2.VideoCapture('./data/videos/ds/13.mov')
    # # # cap = cv2.VideoCapture('./data/videos/DCIM/100MEDIA/DJI_0022.MP4')
    # # from detection import vsplit_ds_frame
    # # ret, frame = cap.read()
    # # img, _ = vsplit_ds_frame(frame, (640, 480))

    wname = "MouseEvent"
    cv2.namedWindow(wname)
    cv2.setMouseCallback(wname, onMouse, [wname, p_info])
    cv2.imshow(wname, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    p_info.points=[[379, 216],
     [435, 233],
     [320, 273],
     [273, 244]]

    p_info.points = np.array(p_info.points).astype(float)
    camera_A = Camera(p_info.points)

    img = cv2.imread('./data/images/wtt2019b.png')
    img = cv2.resize(img, (640, 480))
    p_info = PointInfo(npoints, img)
    cv2.imshow('sourceB', img)

    p_info.points=[[260, 196],
     [378, 197],
     [382, 268],
     [253, 268]]
    p_info.points = np.array(p_info.points).astype(float)
    camera_B = Camera(p_info.points)

    # triangulate ping-pong points

    Ipoint_onimageA = np.array([378.,222.])
    Ipoint_onimageB = np.array([352.,209.])

    w_point_projected = cv2.triangulatePoints(camera_A.projection_matrix, camera_B.projection_matrix, Ipoint_onimageA, Ipoint_onimageB)
    w_point_projected /= w_point_projected[3]

    # plot part
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    x = [TABLE_POINTS[0,0], TABLE_POINTS[1,0], TABLE_POINTS[2,0], TABLE_POINTS[3,0], camera_A.camera_position[0,0], camera_B.camera_position[0,0], w_point_projected[0,0]]
    y = [TABLE_POINTS[0,1], TABLE_POINTS[1,1], TABLE_POINTS[2,1], TABLE_POINTS[3,1], camera_A.camera_position[1,0], camera_B.camera_position[1,0], w_point_projected[1,0]]
    z = [TABLE_POINTS[0,2], TABLE_POINTS[1,2], TABLE_POINTS[2,2], TABLE_POINTS[3,2], camera_A.camera_position[2,0], camera_B.camera_position[2,0], w_point_projected[2,0]]

    max_range = np.array([max(x)-min(x), max(y)-min(y), max(z)-min(z)]).max() * 1.5

    ax.set_xlim(-max_range/2, max_range/2)
    ax.set_ylim(-max_range/2, max_range/2)
    ax.set_zlim(0, max_range)

    ax.scatter(x, y, z)

    import main
    pairs = main.calc_corresponding_points([Ipoint_onimageA], [Ipoint_onimageB], camera_A, camera_B)
    point = main.triangulate(pairs[0][0], pairs[0][1], camera_A, camera_B)


    # -------------calc vec CameraA to point, Camera to point 
    TA = 1500
    w_point_on_line_A = camera_A.camera_position+camera_A.vecs_PoV2objects([Ipoint_onimageA])[0]*TA
    TB = 1000
    w_point_on_line_B = camera_B.camera_position+camera_B.vecs_PoV2objects([Ipoint_onimageB])[0]*TB

    linex = [camera_A.camera_position[0,0], w_point_on_line_A[0]]
    liney = [camera_A.camera_position[1,0], w_point_on_line_A[1]]
    linez = [camera_A.camera_position[2,0], w_point_on_line_A[2]]
    ax.plot(linex, liney, linez, "o-", color="#00aa00", ms=4, mew=0.5)

    linex = [camera_B.camera_position[0,0], w_point_on_line_B[0]]
    liney = [camera_B.camera_position[1,0], w_point_on_line_B[1]]
    linez = [camera_B.camera_position[2,0], w_point_on_line_B[2]]
    ax.plot(linex, liney, linez, "o-", color="#00aa00", ms=4, mew=0.5)


    vecA = camera_A.vecs_PoV2objects([Ipoint_onimageA])[0]
    vecB = camera_B.vecs_PoV2objects([Ipoint_onimageB])[0]
    cross = np.cross(vecA.T,vecB.T)
    cross = cross/np.linalg.norm(cross)

    vecAB = camera_A.camera_position-camera_B.camera_position

    plt.show()
    # plt.pause(.01)
