
import numpy as np
import cv2
import csv
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from settings import Settings


class PingpongPlot():
    def __init__(self, cameras=None):
        self.fig = plt.figure()
        self.ax = Axes3D(self.fig)
        plot_points = []
        self.camera_pos = []
        for i in range(3):
            plot_points.append([Settings.get('TABLE_POINTS')[0,i], Settings.get('TABLE_POINTS')[1,i], Settings.get('TABLE_POINTS')[2,i], Settings.get('TABLE_POINTS')[3,i]])
            if cameras: self.camera_pos.append([cameras[0].camera_position[i,0], cameras[1].camera_position[i,0]])
        self.max_range = np.array([max(plot_points[0])-min(plot_points[0]), max(plot_points[1])-min(plot_points[1]), max(plot_points[2])-min(plot_points[2])]).max() * 1.5

        TABLE_WIDTH = Settings.get('TABLE_WIDTH')
        TABLE_LENGTH = Settings.get('TABLE_LENGTH')
        TABLE_HEIGHT = Settings.get('TABLE_HEIGHT')
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
        # if self.camera_pos: self.ax.scatter(self.camera_pos[0], self.camera_pos[1], self.camera_pos[2])
        plt.pause(.0001)

    def open_writer(self):
        self.file = open('./data/csv/sample_plot.csv', 'w')
        self.writer = csv.writer(self.file)
        self.fps = 1./60.
        self.time = 0
        self.prev = np.array([[0],[0],[0]])

    def close_writer(self):
        self.file.close

    def write(self, points):
        if points:
            self.writer.writerow([self.time, 0, points[0][0,0],points[0][1,0],points[0][2,0],0,0,0,0,0,0])
            self.prev = points[0]
        else:
            self.writer.writerow([self.time, 0])
        self.time = self.time + self.fps


def datastadium_plot(num=3):
    with open('data/csv/20190318_'+str(num)+'.csv') as f:
        reader = csv.reader(f)
        outputter = PingpongPlot()
        prev = np.array([[0],[0],[0]])
        for row in reader:
            if len(row)==11:
                # if np.array_equal(point,prev): print(float(row[0]))
                # else: outputter.plot([point])
                # prev = point
                print(float(row[0]))
                outputter.plot([point])

def mydata_plot(num=3):
    # with open('data/csv/tempmatch_writer_row_'+str(num)+'.csv') as f:
    with open('data/csv/sample_writer_row_'+str(num)+'.csv') as f:
        reader = csv.reader(f)
        outputter = PingpongPlot()
        prev = np.array([[0],[0],[0]])
                # else: outputter.plot([point])
                # prev = point
                outputter.plot([point])

def compare_me_ds_plot(num=3):
    cap = cv2.VideoCapture('./data/videos/ds/1'+str(num)+'.mov')
    for i in range(int(60*5.5)):
        ret, frame = cap.read()
    fps = 1./30.
    time_counter = 0
    time_ds = 0
    time_me = 0
    diff = 2.38 # 2.55
    jump = 23 # 1.2
    seq = {
        'image': [],
        'ds': [],
        'me': [],
        'ts': [],
    }

    with open('data/csv/20190318_'+str(num)+'.csv') as f_ds:
        with open('data/csv/sample_writer_row_'+str(num)+'.csv') as f_me:
            data_ds = [row for row in csv.reader(f_ds)]
            data_me = [row for row in csv.reader(f_me)]
            outputter = PingpongPlot()
            point_me = np.array([[0],[0],[0]])
            point_ds = np.array([[0],[0],[0]])
            while data_ds:
                row_ds = data_ds.pop(0)
                if len(row_ds)!=11: continue
    time_1 = 0
    time_2 = 0
    diff = 0.1
    jump = 0

    with open(csv_1) as f_1:
        data_1 = [row for row in csv.reader(f_1)]
    with open(csv_2) as f_2:
        data_2 = [row for row in csv.reader(f_2)]
    outputter = PingpongPlot()
    point_1 = np.array([[0],[0],[0]])
    point_2 = np.array([[0],[0],[0]])

    while data_1:
        row_1 = data_1.pop(0)
        if len(row_1)!=11: continue
        time_1 = float(row_1[0])-diff-jump
        break

    while data_2 and ret:
        row_2 = data_2.pop(0)
        ret, frame = cap.read()
        if len(row_2)!=11: continue
        time_2 = float(row_2[0])-jump
        break

    while data_1 and data_2 and ret:
        while data_1:
            if time_1 > time_counter: break
            if len(row_1)==11:
                point_1 = np.array([[float(row_1[2])], [float(row_1[3])], [float(row_1[4])]])
            row_1 = data_1.pop(0)
            if len(row_1)!=11: continue
            time_1 = float(row_1[0])-diff-jump
            
        while data_2 and ret:
            if time_2 > time_counter: break
            if len(row_2)==11: 
                point_2 = np.array([[float(row_2[2])], [float(row_2[3])], [float(row_2[4])]])
            row_2 = data_2.pop(0)
            ret, frame = cap.read()
            time_2 = float(row_2[0])-jump
        
        outputter.plot([point_2])
        cv2.imshow("frame", frame)
        time_counter += fps