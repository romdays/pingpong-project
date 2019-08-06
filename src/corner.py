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
