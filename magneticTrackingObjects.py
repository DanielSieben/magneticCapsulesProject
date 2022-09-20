import math
import numpy as np


# TODO: error catching on all of below (dimensions, + values, etc.)
# TODO add constrained motion in magneticCapsule
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


class pointDipole:
    # point magnet at some pose, generates a magnetic field at any cartesian point in the x, y, and z directions
    # according to its pose. field is assumed to be symmetrical about the z-axis.
    def __init__(self,
                 x: float = 0., y: float = 0., z: float = 0.,
                 angleX: float = 0., angleY: float = 0.,
                 Br: float = 1.4, volume: float = 1.25e-7):
        #   inputs:
        #   x = x-axis location, m (float)
        #   y = y-axis location, m (float)
        #   z = z-axis location, m (float)
        #   angle_x = angle relative to x-axis, rad (float)
        #   angle_y = angle relative to y-axis, rad (float)
        #   Br = magnetic remanence of the point magnet, T (float)
        #   volume = point magnet volume, m^3 (float)
        self.pose = np.array([x, y, z, angleX, angleY])
        mu0 = 4 * np.pi * 10e-2 # vacuum magnetic permeability, Henry / meter
        self.mBar = Br * volume / mu0

    def generateDipoleField(self, x: float, y: float, z: float):
        # generate magnetic field at a cartesian x, y, z point according to the current pose
        #   inputs:
        #   x = x point of the field, m (float)
        #   y = y point of the field, m (float)
        #   z = z point of the field, m (float)
        #   outputs:
        #   Bfield = input point magnetic field strength in the x, y, and z directions, T (3x0 numpy ndarray.float)
        # repeated calculations
        xbar = x - self.pose[0]
        ybar = y - self.pose[1]
        zbar = z - self.pose[2]
        r = max(math.sqrt(xbar ** 2 + ybar ** 2 + zbar ** 2), 1e-9)
        rm1p5 = r ** -1.5
        sxcy = math.sin(self.pose[3]) * math.cos(self.pose[4])
        sxsy = math.sin(self.pose[3]) * math.sin(self.pose[4])
        cx = math.cos(self.pose[3])
        term1 = 3 * (sxcy * xbar + sxsy * ybar + cx * zbar) * r ** -2.5
        # field calculations
        # slower numpy implementation removed (~one quarter as fast as direct calculation)
        # moment = self.mBar * np.array([sxcy, sxsy, cx])
        # distanceVector = np.array([x - self.pose[0], y - self.pose[1], z - self.pose[2]])
        # r = math.sqrt(sum(distanceVector ** 2))
        # return 3 * distanceVector.dot(moment.dot(distanceVector)) / r ** 5 - moment / r ** 3
        Bx = self.mBar * (xbar * term1 - sxcy * rm1p5)
        By = self.mBar * (ybar * term1 - sxsy * rm1p5)
        Bz = self.mBar * (zbar * term1 - cx * rm1p5)
        return np.array([Bx, By, Bz])


class magneticCapsule:
    def __init__(self, magnets):
        self.magnets = magnets
        self.magnetPoses = np.array([magnet.pose for magnet in self.magnets]).flatten()

    def generateDipoleField(self, x, y, z):
        B = np.zeros(3)
        for magnet in self.magnets:
            B += magnet.generateDipoleField(x, y, z)
        return B

    def updatePose(self, newPose):
        for idx, magnet in enumerate(self.magnets):
            magnet.pose = newPose[idx * 5: idx * 5 + 5]
        self.magnetPoses[:] = newPose


class magneticSensor:
    # magnetic sensor which measures the magnetic field (Bx, By, Bz) in T
    def __init__(self,
                 x: float, y: float, z: float,
                 angleX: float, angleY: float, angleZ: float,
                 sensorRangeLimits: (float, float) = None,
                 sensorStdDeviation: float = 0.,
                 magnetObject=None):
        #   inputs:
        #   x = x-axis location, m (float)
        #   y = y-axis location, m (float)
        #   z = z-axis location, m (float)
        #   angleX = angle relative to x-axis, rad (float)
        #   angleY = angle relative to y-axis, rad (float)
        #   angleZ = angle relative to z-axis, rad (float)
        #   sensorRangeLimits = sensor range limits, T (2-tuple of (float, float))
        #   sensorStdDeviation = sensor standard deviations, T (float)
        self.pose = np.array([x, y, z, angleX, angleY, angleZ])
        self.rotationMatrix = rotationMatrix(angleX, angleY, angleZ)
        self.sensorRangeLimits = sensorRangeLimits
        self.sensorStdDeviation = sensorStdDeviation
        self.currentMeasurement = np.zeros(3)
        self.magnetObject = magnetObject

    def measure(self, magnetObject, noise=False):
        #   inputs:
        #   magnetObject = magnet object with the method "generateDipoleField(x, y, z)" which returns the magnetic
        #   field at the x, y, z point passed in (Bx, By, Bz) format, T
        #   noise = boolean of whether to apply zero mean gaussian noise per sensorStdDeviation
        #   outputs:
        #   currentMeasurement = current magnetic field measurement (Bx, By, Bz), T (3x0 numpy ndarray)
        self.currentMeasurement[:] = self.rotationMatrix.dot(magnetObject.generateDipoleField(*self.pose[:3]))
        if noise:
            self.currentMeasurement += np.random.normal(0.0, self.sensorStdDeviation, size=3)
        if self.sensorRangeLimits is not None:
            self.currentMeasurement[:] = np.maximum(self.sensorRangeLimits[0], np.minimum(self.sensorRangeLimits[1],
                                                    self.currentMeasurement))
        return self.currentMeasurement

    def measureState(self, state):
        if self.magnetObject is None:
            return np.full(3, np.nan)
        for idx, magnet in enumerate(self.magnetObject.magnets):
            magnet.pose = state[idx * 5: idx * 5 + 5]
        return self.measure(self.magnetObject)

    def updatePose(self, newPose):
        self.pose = newPose
        self.rotationMatrix = rotationMatrix(*newPose[3:])


class sensorArray:
    def __init__(self, sensors, magnetObject=None):
        self.sensors = sensors
        self.sensorPoses = np.array([sensor.pose for sensor in self.sensors]).flatten()
        self.currentMeasurement = np.array([sensor.currentMeasurement for sensor in sensors]).flatten()
        self.magnetObject = magnetObject

    def measure(self, magnetObject, noise=False):
        for idx, sensor in enumerate(self.sensors):
            self.currentMeasurement[idx * 3: idx * 3 + 3] = sensor.measure(magnetObject, noise=noise)
        return self.currentMeasurement

    def measureState(self, state):
        if self.magnetObject is None:
            return np.full(3 * len(self.sensors), np.nan)
        for idx, magnet in enumerate(self.magnetObject.magnets):
            magnet.pose = state[idx * 5: idx * 5 + 5]
        return self.measure(self.magnetObject)

    def updateSensorPoses(self, newPoses):
        for idx, sensor in enumerate(self.sensors):
            sensor.updatePose(newPoses[idx * 6: idx * 6 + 6])
            self.sensorPoses[idx * 6: idx * 6 + 6] = sensor.pose


def createLyleArray():
    nX = 3
    nY = 3
    spacingX = 0.05
    spacingY = 0.05
    minRange = -4912e-1 # T
    maxRange = 4912e-1 # T
    stdDev = 1e-6 # T
    sensors = []
    for x in range(nX):
        for y in range(nY):
            sensors.append(magneticSensor(x * spacingX, y * spacingY, 0.,
                                          0., 0., 0.,
                                          (minRange, maxRange),
                                          stdDev))
    return sensorArray(sensors)


def createHydrogelCapsule():
    remanence = 1.4
    vol = 1.25e-7
    poses = np.array([[0, 0, 0, 0, 0, 0],
                      [0, 0, 0.1, np.pi * 0.5, 0, 0]])
    magnets = []
    for pose in poses:
        magnets.append(pointDipole(*pose, Br=remanence, volume=vol))
    return magneticCapsule(magnets)
