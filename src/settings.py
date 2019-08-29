import numpy as np

class Settings():
    TABLE_WIDTH = 1.525 # m
    TABLE_LENGTH = 2.740 # m
    TABLE_HEIGHT = 0.760
    
    __settings = {
        'CALIB_PARAM': np.load('./data/npz/calib.npz'),

        'FRAME_INTERVAL': 1,

        'TABLE_WIDTH': TABLE_WIDTH,
        'TABLE_LENGTH': TABLE_LENGTH,
        'TABLE_HEIGHT': TABLE_HEIGHT,

        'TABLE_POINTS': np.array([
            (-TABLE_LENGTH/2,  TABLE_WIDTH/2, TABLE_HEIGHT),
            (-TABLE_LENGTH/2, -TABLE_WIDTH/2, TABLE_HEIGHT),
            ( TABLE_LENGTH/2, -TABLE_WIDTH/2, TABLE_HEIGHT),
            ( TABLE_LENGTH/2,  TABLE_WIDTH/2, TABLE_HEIGHT),
            ]),

        'GAMMA': 1.3,
        'LOWER_COLOR': np.array([0,0,50]),
        'UPPER_COLOR': np.array([10,50,255]),

        'MIN_CIRCULARITY': 0.75,
        'MIN_CONTOUR_AREA': None,
        'MAX_CONTOUR_AREA': None,

        'MAX_DISTANCE': 0.10, # m,

        'TEMPLATE_MIN_CIRCULARITY': 0.95,
        'TEMPLATE_MIN_SIMILARITY': 0.85,
        }
    
    def __init__(self, image_shape):
        pixels = image_shape[0]*image_shape[1]
        self.update('MIN_CONTOUR_AREA', pixels/(11.5**4))
        self.update('MAX_CONTOUR_AREA', pixels/(8.**4))

    @classmethod
    def get(cls, key):
        return cls.__settings.get(key)

    @classmethod
    def update(cls, key, value):
        cls.__settings[key] = value


    @classmethod
    def get_template(cls, key):
        if cls.__settings.get('TEMPLATE_'+key) is None and cls.__settings.get('FIRST_TEMPLATE_IMAGE'):
            return cls.__settings.get('FIRST_TEMPLATE_IMAGE')
        return cls.__settings.get('TEMPLATE_'+key)

    @classmethod
    def update_template(cls, key, value):
        if cls.__settings.get('TEMPLATE_'+key) is None:
            cls.__settings['TEMPLATE_'+key] = []
            cls.__settings['FIRST_TEMPLATE_IMAGE'] = [value]
        cls.__settings['TEMPLATE_'+key].append(value)

    @classmethod
    def remove_template(cls, key, index):
        if cls.__settings.get('TEMPLATE_'+key) is None: return
        elif len(cls.__settings.get('TEMPLATE_'+key)) > index:
            cls.__settings.get('TEMPLATE_'+key).pop(index)

