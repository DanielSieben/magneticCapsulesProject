import matplotlib.pyplot as plt
import numpy as np
import math


# TODO error catching for data types and dimensions on below
# TODO add multiprocessing options for data updates (pathos.multiprocessing), update speed is adequate for now at ~2Hz
# TODO add function to create a display from a sensor array and capsule object pair (magneticTrackingObjects)
# TODO remove the rotation / htm functions to their own common file
# TODO autoscaling of 3D axes limits based on pose history, not included for now as limits are set by physical means
def make2DLinePlots(nPlots: int, nLines: int, lineLabels=None, axisLabels=None):
    closestSquare = np.ceil(nPlots ** 0.5).astype('int')
    fig, axs = plt.subplots(closestSquare, closestSquare)
    for _ in range(closestSquare**2 - nPlots): # remove spare axes from square display
        axs = np.delete(axs, -1)
    fig.tight_layout = True
    fig.canvas.manager.set_window_title('Raw Sensor Data')
    axLines = []
    for axN, ax in enumerate(axs.ravel()):
        axLines.append(ax.plot(0, np.zeros((1, nLines))))
        # legend on the first plot only
        if lineLabels is not None and axN == 0:
            ax.legend(lineLabels, loc='upper left')
        # x label on the first plot of the bottom row
        if axisLabels is not None and axN == closestSquare * (closestSquare - 1):
            ax.set_xlabel(axisLabels[0])
        # y label on the middle-most plot of the first column
        if axisLabels is not None and axN == closestSquare * (closestSquare - np.ceil(closestSquare/2)):
            ax.set_ylabel(axisLabels[1])
    return fig, axs, axLines


def make3DPlot(nMagnets: int, nSensors: int, nPredMag: int = 0, nPredSen: int = 0, axisLabels=None):
    fig = plt.figure('3D Tracking')
    ax = fig.add_subplot(111, projection='3d')
    lines = {'actual': [], 'predicted': []}
    magnets = {'actual': [], 'predicted': []}
    sensors = {'actual': [], 'predicted': []}
    for _ in range(nMagnets): # red = magnetic N, blue = magnetic S
        lines['actual'].append(ax.plot([], [], [], color='r', alpha=0.25)[0])
        magnets['actual'].append([ax.plot([], [], [], color='r', linewidth=2)[0],
                                  ax.plot([], [], [], color='b', linewidth=2)[0]])
    for _ in range(nPredMag): # red = magnetic N, blue = magnetic S
        lines['predicted'].append(ax.plot([], [], [], color='b', alpha=0.1)[0])
        magnets['predicted'].append([ax.plot([], [], [], color='r', linewidth=2, alpha=0.33)[0],
                                     ax.plot([], [], [], color='b', linewidth=2, alpha=0.33)[0]])
    for _ in range(nSensors): # plotted as triangles, height = x-direction, base = y-direction, normal=z-direction
        sensors['actual'].append([ax.plot([], [], [], color='k', linewidth=3, alpha=0.5)[0],
                                  ax.plot([], [], [], color='k', linewidth=3, alpha=0.5)[0],
                                  ax.plot([], [], [], color='k', linewidth=3, alpha=0.5)[0]])
    for _ in range(nPredSen): # plotted as triangles, height = x-direction, base = y-direction, normal=z-direction
        sensors['predicted'].append([ax.plot([], [], [], color='g', linewidth=3, alpha=0.25)[0],
                                     ax.plot([], [], [], color='g', linewidth=3, alpha=0.25)[0],
                                     ax.plot([], [], [], color='g', linewidth=3, alpha=0.25)[0]])
    if axisLabels is not None:
        ax.set_xlabel(axisLabels[0])
        ax.set_ylabel(axisLabels[1])
        ax.set_zlabel(axisLabels[2])
    return fig, ax, lines, magnets, sensors


def rotationMatrix(angleX: float, angleY: float, angleZ: float):
    #   inputs:
    #   angleX = angle relative to x-axis, rad (float)
    #   angleY = angle relative to y-axis, rad (float)
    #   angleZ = angle relative to z-axis, rad (float)
    #   outputs:
    #   rotationMatrix = 3d rotation matrix (3x3 numpy ndarray.float)
    # repeated calculations
    cx = math.cos(angleX)
    sx = math.sin(angleX)
    cy = math.cos(angleY)
    sy = math.sin(angleY)
    cz = math.cos(angleZ)
    sz = math.sin(angleZ)
    # matrix calculation
    # slower matrix multiplication method removed (~half as fast as direct calculation)
    # Rx = np.array([[cx, -sx, 0], [sx, cx, 0], [0, 0, 1]])
    # Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    # Rz = np.array([[1, 0, 0], [0, cz, -sz], [0, sz, cz]])
    # return Rx.dot(Ry).dot(Rz)
    return np.array([
        [cx*cy, cx*sy*sz - sx*cz, cx*sy*cz + sx*sz],
        [sx*cy, sx*sy*sz + cx*cz, sx*sy*cz - cx*sz],
        [-sy, cy*sz, cy*cz]
    ])


def homTransformationMatrix(dx: float, dy: float, dz: float, dax: float, day: float, daz: float):
    #   inputs:
    #   dx = change in x-axis direction, m (float)
    #   dy = change in y-axis direction, m (float)
    #   dz = change in z-axis direction, m (float)
    #   dax = change in angle relative to x-axis, rad (float)
    #   day = change in angle relative to y-axis, rad (float)
    #   daz = change in angle relative to z-axis, rad (float)
    #   outputs:
    #   htm = homogenous transformation matrix (4x4 numpy ndarray.float)
    htm = np.zeros((4, 4))
    htm[:3, :3] = rotationMatrix(dax, day, daz)
    htm[:3, -1] = [dx, dy, dz]
    htm[-1, -1] = 1
    return htm


class magnetLocalizationDisplay:
    def __init__(self,
                 nMagnets: int, nSensors: int,
                 nPredMagnets: int = 0, nPredSensors: int = 0,
                 historySize3D: int = 10, historySizeRawData: int = 100):
        # Magnets use a 5-state pose (x, y, z, aX, aY)
        self.magPoseHist = {'actual': np.full((historySize3D, 5 * nMagnets), np.nan),
                            'predicted': np.full((historySize3D, 5 * nPredMagnets), np.nan)}
        # Sensors use a 6-state pose (x, y, z, aX, aY, aZ)
        # TODO: lines for showing sensor movement over time not implemented at the moment, static array assumed for now
        self.senPoseHist = {'actual': np.full((historySize3D, 6 * nSensors), np.nan),
                            'predicted': np.full((historySize3D, 6 * nPredSensors), np.nan)}
        # Raw data readings from sensors, 3D magnetic field
        self.rawDataHist = {'data': np.full((historySizeRawData, nSensors * 3), np.nan),
                            'time': np.full(historySizeRawData, np.nan)}
        self.rawDataFig, self.rawDataAx, self.rawDataLines = \
            make2DLinePlots(nSensors, 3, ['Bx', 'By', 'Bz'], ['Time, s', 'Magnetic Field, T'])
        self.fig3D, self.ax3D, self.lines3D, self.magnets3D, self.sensors3D = \
            make3DPlot(nMagnets, nSensors, nPredMagnets, nPredSensors, axisLabels=['x (meters)', 'y', 'z'])
        self.disp3DRad = 0.005 # unit length for 3D display magnets / sensors
        self.ax3D.set_xlim3d((0, 0.1))
        self.ax3D.set_ylim3d((0, 0.1))
        self.ax3D.set_zlim3d((-0.01, 0.2))
        plt.show(block=False)

    def __del__(self):
        plt.close(self.rawDataFig)
        plt.close(self.fig3D)

    def updateRawData(self, newData, timeStamp):
        self.rawDataHist['data'][:-1, :] = self.rawDataHist['data'][1:, :]
        self.rawDataHist['data'][-1, :] = newData
        self.rawDataHist['time'][:-1] = self.rawDataHist['time'][1:]
        self.rawDataHist['time'][-1] = timeStamp
        count = 0
        for axN, ax in enumerate(self.rawDataAx.ravel()):
            for line in self.rawDataLines[axN]:
                line.set_data(self.rawDataHist['time'], self.rawDataHist['data'][:, count])
                count += 1
            ax.relim()
            ax.autoscale_view(True, True, True)
        self.rawDataFig.canvas.draw()
        self.rawDataFig.canvas.flush_events()

    # TODO double check if spherical coordinates here is faster than htm implementation
    def update3DData(self, newPoses, poseType='actual'):
        self.magPoseHist[poseType][:-1, :] = self.magPoseHist[poseType][1:, :]
        self.magPoseHist[poseType][-1] = newPoses
        for magN, mag in enumerate(self.magnets3D[poseType]):
            index = 5 * magN
            self.lines3D[poseType][magN].set_data(self.magPoseHist[poseType][:, index],
                                                  self.magPoseHist[poseType][:, index + 1])
            self.lines3D[poseType][magN].set_3d_properties(self.magPoseHist[poseType][:, index + 2])
            xPlus = self.disp3DRad * math.cos(newPoses[index + 3]) * math.sin(newPoses[index + 4])
            yPlus = self.disp3DRad * math.sin(newPoses[index + 3]) * math.sin(newPoses[index + 4])
            zPlus = self.disp3DRad * math.cos(newPoses[index + 4])
            mag[0].set_data([newPoses[index], newPoses[index] + xPlus],
                            [newPoses[index + 1], newPoses[index + 1] + yPlus])
            mag[0].set_3d_properties([newPoses[index + 2], newPoses[index + 2] + zPlus])
            mag[1].set_data([newPoses[index], newPoses[index] - xPlus],
                            [newPoses[index + 1], newPoses[index + 1] - yPlus])
            mag[1].set_3d_properties([newPoses[index + 2], newPoses[index + 2] - zPlus])
        self.fig3D.canvas.draw()
        self.fig3D.canvas.flush_events()

    def updateSensorData(self, newPoses, poseType='actual'):
        self.senPoseHist[poseType][:-1, :] = self.senPoseHist[poseType][1:, :]
        self.senPoseHist[poseType][-1, :] = newPoses
        point1 = np.array([0.5 * self.disp3DRad, 0, 0, 1])
        point2 = np.array([-0.5 * self.disp3DRad, 0.25 * self.disp3DRad, 0, 1])
        point3 = np.array([-0.5 * self.disp3DRad, -0.25 * self.disp3DRad, 0, 1])
        for senN, sen in enumerate(self.sensors3D[poseType]):
            htm = homTransformationMatrix(*newPoses[senN * 6: senN * 6 + 6])
            point1p = htm.dot(point1)[:-1]
            point2p = htm.dot(point2)[:-1]
            point3p = htm.dot(point3)[:-1]
            sen[0].set_data([point1p[0], point2p[0]], [point1p[1], point2p[1]])
            sen[0].set_3d_properties([point1p[2], point2p[2]])
            sen[1].set_data([point2p[0], point3p[0]], [point2p[1], point3p[1]])
            sen[1].set_3d_properties([point2p[2], point3p[2]])
            sen[2].set_data([point3p[0], point1p[0]], [point3p[1], point1p[1]])
            sen[2].set_3d_properties([point3p[2], point1p[2]])
        self.fig3D.canvas.draw()
        self.fig3D.canvas.flush_events()

    def update3DAxes(self, newAxisX, newAxisY, newAxisZ):
        if newAxisX is not None:
            self.ax3D.set_xlim3d(newAxisX)
        if newAxisY is not None:
            self.ax3D.set_ylim3d(newAxisY)
        if newAxisZ is not None:
            self.ax3D.set_zlim3d(newAxisZ)

    def update3DUnitLength(self, newSize: float):
        self.disp3DRad = newSize
        self.update3DData(self.magPoseHist[-1, :])
        self.updateSensorData(self.senPoseHist[-1, :])
        if len(self.magnets3D['predicted']) != 0:
            self.update3DData(self.magPoseHist[-1, :], 'predicted')
        if len(self.sensors3D['predicted']) != 0:
            self.updateSensorData(self.senPoseHist[-1, :], 'predicted')


if __name__ == '__main__': # example display
    disp = magnetLocalizationDisplay(1, 9, 1, 9)
    sensorPoses = []
    predSensorPoses = []
    for nx in range(3):
        for ny in range(3):
            if nx == 1 and ny == 1:
                sensorPoses.append([nx * 0.05, ny * 0.05, 0.01, np.pi/4, 0, 0])
                predSensorPoses.append([nx * 0.05, ny * 0.05, 0, np.pi/6, 0, 0])
                continue
            sensorPoses.append([nx * 0.05, ny * 0.05, 0, 0, 0, 0])
            predSensorPoses.append([nx * 0.05, ny * 0.05, -0.005, *np.random.uniform(-np.pi, np.pi, size=3)])
    sensorPoses = np.array(sensorPoses).flatten()
    disp.updateSensorData(sensorPoses)
    predSensorPoses = np.array(predSensorPoses).flatten()
    disp.updateSensorData(predSensorPoses, 'predicted')
    for x in range(200):
        newData = np.array([math.cos(x*0.1), math.sin(x*0.1), math.sin(x*0.1 + np.pi/8)] * 9) \
                  + np.random.normal(0, 0.5, size=3 * 9)
        disp.updateRawData(newData, x)
        newPose = np.array([0.05 * math.cos(x * 0.25) + 0.05,
                            0.05 * math.sin(x * 0.25) + 0.05,
                            x * 0.0005,
                            np.pi / 2,
                            x * 0.1])
        disp.update3DData(newPose)
        newPredPose = newPose + np.random.normal(0, 0.01, size=5)
        disp.update3DData(newPredPose, 'predicted')
    plt.show()
