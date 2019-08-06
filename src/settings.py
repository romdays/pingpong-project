import numpy as np

class Settings():
    TABLE_WIDTH = 152.5 # cm
    TABLE_LENGTH = 274.0 # cm
    
    __settings = {
        'CALIB_PARAM': np.load('./data/npz/calib.npz'),

        'FRAME_INTERVAL': 1,

        'TABLE_POINTS': np.array([
            (-TABLE_LENGTH/2,  TABLE_WIDTH/2, 0),
            (-TABLE_LENGTH/2, -TABLE_WIDTH/2, 0),
            ( TABLE_LENGTH/2, -TABLE_WIDTH/2, 0),
            ( TABLE_LENGTH/2,  TABLE_WIDTH/2, 0),
            ]),

        'LOWER_COLOR': np.array([0,0,100]),
        'UPPER_COLOR': np.array([10,50,255]),

        'MIN_CIRCULARITY': 0.8,
        'MIN_COUNTOUR_AREA': 20,
        'MAX_COUNTOUR_AREA': 50,

        'MAX_DISTANCE': 15, # cm

        }

    @classmethod
    def get(cls, key):
        return cls.__settings[key]

    @classmethod
    def update(cls, key, value):
        cls.__settings[key] = value
