import numpy as np
import cv2 

from settings import (
    CALIB_PARAM,
    TABLE_POINTS,
)

class Camera():
    def __init__(self, points_on_image):
        CALIB_PARAM = np.load('./data/npz/calib.npz')
        self.camera_matrix = CALIB_PARAM['cameraMatrix']
        self.dist_coeffs = CALIB_PARAM['distCoeffs']

        points_on_image = np.asarray(points_on_image).astype(float)

        ret, self.rvecs, self.tvecs = cv2.solvePnP(TABLE_POINTS, points_on_image, self.camera_matrix, self.dist_coeffs)
        # self.rvecs = np.array((self.rvecs[0,0],self.rvecs[1,0],self.rvecs[2,0]))

        self.rmat = cv2.Rodrigues(self.rvecs)[0]
        self.camera_position = -np.matrix(self.rmat).T * np.matrix(self.tvecs)
        self.projection_matrix = np.matmul(self.camera_matrix, np.hstack((self.rmat, self.tvecs)))

    def _convert_point_screen2world(self, screen_point):
        # Pc = RPx + t -> Px = R^(-1)(Pc-t)
        world_point_onscreen = np.array([[(screen_point[0]-self.camera_matrix[0,2])/self.camera_matrix[0,0]-self.tvecs[0,0]], 
                                         [(screen_point[1]-self.camera_matrix[1,2])/self.camera_matrix[1,1]-self.tvecs[1,0]], 
                                         [1-self.tvecs[2,0]]])
        world_point_onscreen = np.matmul(np.linalg.inv(self.rmat),world_point_onscreen)

        return world_point_onscreen
        
    def vecs_PoV2objects(self, screen_points):
        vecs = []
        for screen_point in screen_points:
            world_point = self._convert_point_screen2world(screen_point)
            vecs.append(world_point-self.camera_position)

        return vecs
        