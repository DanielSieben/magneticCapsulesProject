import magneticTrackingObjects
import magnetLocalizationDisplay
import modelBasedEstimator
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import copy


# time variables
maxTime = 60
dt = 0.5
simTime = np.arange(0, maxTime, dt)

# magnet variables
nMagnets = 1
poseX = np.linspace(0, 0.01, len(simTime))
poseY = np.linspace(0.01, 0, len(simTime))
poseZ = np.linspace(0.02, 0.021, len(simTime))
poseA1 = np.linspace(0, np.pi, len(simTime))
poseA2 = np.linspace(np.pi * 2, 0, len(simTime))
poses = np.vstack((poseX, poseY, poseZ, poseA1, poseA2)).T
magnets = [magneticTrackingObjects.pointDipole(*poses[0, :])]
capsule = magneticTrackingObjects.magneticCapsule(magnets)

# sensor variables
sensorArray = magneticTrackingObjects.createLyleArray()
sensorArray.magnetObject = copy.deepcopy(capsule)

# estimator variables
stateBounds = (np.array([0., 0., 0., 0., 0.]), np.array([0.1, 0.1, 0.1, 2*np.pi, 2*np.pi]))
initEstimate = np.random.uniform(stateBounds[0], stateBounds[1])
measFcn = sensorArray.measureState
estimator = modelBasedEstimator.lsq_modelBasedEstimator(initEstimate, measFcn, stateBounds=stateBounds)

# simulation
meas = np.zeros((len(simTime), len(sensorArray.currentMeasurement)))
estimates = np.zeros((len(simTime), len(initEstimate)))
for idx, t in enumerate(simTime):
    print('t = %.2f of %.2f\r' % (t, maxTime))
    capsule.updatePose(poses[idx, :])
    meas[idx, :] = sensorArray.measure(capsule, noise=True)
    estimates[idx, :] = estimator.estimate(meas[idx, :])

plt.figure('xyz errors')
plt.plot(simTime, poses[:, :3] - estimates[:, :3])
plt.legend(['x', 'y', 'z'])
plt.figure('angle errors')
plt.plot(simTime, poses[:, 3:] - estimates[:, 3:])
plt.legend(['angle x', 'angle y'])
plt.figure('sensor readings')
plt.plot(simTime, meas)
plt.show()
