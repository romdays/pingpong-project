
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from settings import Settings


class PingpongPlot():
    def __init__(self):
        self.fig = plt.figure()
        self.ax = Axes3D(self.fig)
        plot_points = []
        for i in range(3):
            plot_points.append([Settings.get('TABLE_POINTS')[0,i], Settings.get('TABLE_POINTS')[1,i], Settings.get('TABLE_POINTS')[2,i], Settings.get('TABLE_POINTS')[3,i]])#, cameras[0].camera_position[i,0], cameras[1].camera_position[i,0]]
        self.max_range = np.array([max(plot_points[0])-min(plot_points[0]), max(plot_points[1])-min(plot_points[1]), max(plot_points[2])-min(plot_points[2])]).max() * 1.5

        TABLE_WIDTH = 152.5 # cm
        TABLE_LENGTH = 274.0 # cm
        TABLE_HEIGHT = 76.0
        # table
        x = np.linspace(-TABLE_LENGTH/2,TABLE_LENGTH/2,11)
        y = np.linspace(-TABLE_WIDTH/2,TABLE_WIDTH/2,11)
        z = TABLE_HEIGHT
        self.X1,self.Y1 = np.meshgrid(x,y)
        self.Z1 = np.array([[z]]*self.X1.shape[0])
        # net 
        x = 0
        y = np.linspace(-TABLE_WIDTH/1.9,TABLE_WIDTH/1.9,11)
        z = np.linspace(TABLE_HEIGHT,TABLE_HEIGHT+TABLE_WIDTH/10,11)
        self.Y2,self.Z2 = np.meshgrid(y,z)
        self.X2 = np.array([[x]]*self.Y2.shape[0])

    def plot(self, points):
        self.ax.cla()
        self.ax.set_xlim(-self.max_range/2, self.max_range/2)
        self.ax.set_ylim(-self.max_range/2, self.max_range/2)
        self.ax.set_zlim(0, self.max_range)
        # self.ax.scatter(plot_points[0],plot_points[1],plot_points[2])
        self.ax.plot_surface(self.X1,self.Y1,self.Z1,alpha=0.3)
        self.ax.plot_surface(self.X2,self.Y2,self.Z2,alpha=0.3)
        for p in points:
            self.ax.scatter(p[0,0],p[1,0],p[2,0])

        plt.pause(.01)