import numpy as np

class Settings():
    TABLE_WIDTH = 152.5 # cm
    TABLE_LENGTH = 274.0 # cm
    TABLE_HEIGHT = 76.0
    
    __settings = {
        'CALIB_PARAM': np.load('./data/npz/calib.npz'),

        'FRAME_INTERVAL': 1,

        'TABLE_POINTS': np.array([
            (-TABLE_LENGTH/2,  TABLE_WIDTH/2, TABLE_HEIGHT),
            (-TABLE_LENGTH/2, -TABLE_WIDTH/2, TABLE_HEIGHT),
            ( TABLE_LENGTH/2, -TABLE_WIDTH/2, TABLE_HEIGHT),
            ( TABLE_LENGTH/2,  TABLE_WIDTH/2, TABLE_HEIGHT),
            ]),

        'LOWER_COLOR': np.array([0,0,50]),
        'UPPER_COLOR': np.array([10,50,255]),

        'MIN_CIRCULARITY': 0.75,
        'MIN_CONTOUR_AREA': None,
        'MAX_CONTOUR_AREA': None,

        'MAX_DISTANCE': 30, # cm

        }
    
    def __init__(self, image_shape):
        pixels = image_shape[0]*image_shape[1]
        self.update('MIN_CONTOUR_AREA', pixels/(11.5**4))
        self.update('MAX_CONTOUR_AREA', pixels/(8.2**4))

    @classmethod
    def get(cls, key):
        return cls.__settings[key]

    @classmethod
    def update(cls, key, value):
        cls.__settings[key] = value

