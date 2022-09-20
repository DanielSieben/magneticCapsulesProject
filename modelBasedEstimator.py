import numpy as np
import scipy.optimize as so
from pathos.multiprocessing import Pool
from functools import partial
import time


def jacobian2Point(x, fcn, *args, **kwargs):
    dx = 1e-8
    n = len(x)
    fx = fcn(x, *args, **kwargs)
    m = len(fx)
    fxPlus = np.zeros((n, m))
    Dx = x * dx
    Dx[Dx == 0] = dx
    xPlusMtx = np.full((n, n), x) + np.diag(Dx)
    for idx, xPlus in enumerate(xPlusMtx):
        fxPlus[idx, :] = fcn(xPlus, *args, **kwargs)
    return (fxPlus - fx).T/Dx # two-point numerical derivative


def mp_jacobian2Point(x, fcn, pool, *args, **kwargs):
    # multiprocess parallel evaluation included for speed
    dx = 1e-8
    n = len(x)
    fx = fcn(x, *args, **kwargs)
    Dx = x * dx
    Dx[Dx == 0] = dx
    xPlusMtx = np.full((n, n), x) + np.diag(Dx)
    fxPlus = np.array(pool.map(fcn, xPlusMtx))
    return (fxPlus - fx).T/Dx # two-point numerical derivative


def randomWalk(state, covariance, dt, *args, **kwargs):
    return state + np.random.multivariate_normal(np.zeros(state.shape), covariance) * dt


class lsq_modelBasedEstimator:
    def __init__(self,
                 initialEstimate, measFcn, measCov=None, measJacobian=None, measBounds=None,
                 stateFcn=None, stateCov=None, stateJacobian=None, stateBounds=None,
                 optimizerAlgorithm='lm'):
        # defaults and error handling
        # self.pool = Pool() # pool for multiprocessing
        defaultMeasJacobian = partial(jacobian2Point, fcn=measFcn) if optimizerAlgorithm == 'kalman' else '2-point' # '2-point', '3-point', 'sc', f(x, *args, *kwargs)
        defaultMeasCov = np.diag(0.05 * measFcn(initialEstimate)) # 5% of initial state measurement
        defaultMeasCov[range(len(defaultMeasCov)), range(len(defaultMeasCov))] = np.maximum(1e-6, defaultMeasCov[range(len(defaultMeasCov)), range(len(defaultMeasCov))])
        defaultMeasBounds = (-np.inf, np.inf)
        defaultStateCov = np.diag(0.05 * initialEstimate) # 5% of initial state
        defaultStateCov[range(len(defaultStateCov)), range(len(defaultStateCov))] = np.maximum(1e-6, defaultStateCov[
            range(len(defaultStateCov)), range(len(defaultStateCov))])
        defaultStateFcn = partial(randomWalk, covariance=stateCov if stateCov is not None else defaultStateCov)
        defaultStateJacobian = partial(jacobian2Point, fcn=defaultStateFcn) if optimizerAlgorithm == 'kalman' else '2-point'
        defaultStateBounds = (-np.inf, np.inf)
        # attributes
        self.curEstimate = initialEstimate
        self.measFcn = measFcn # f(x, *args, **kwargs)
        self.curMeas = self.measFcn(initialEstimate)
        self.measJacobian = measJacobian if measJacobian is not None else defaultMeasJacobian
        self.measCov = measCov if measCov is not None else defaultMeasCov
        self.measBounds = measBounds if measBounds is not None else defaultMeasBounds
        self.stateCov = stateCov if stateCov is not None else defaultStateCov
        self.stateFcn = stateFcn if stateFcn is not None else defaultStateFcn
        self.stateJacobian = stateJacobian if stateJacobian is not None else defaultStateJacobian
        self.stateBounds = stateBounds if stateBounds is not None else defaultStateBounds
        self.optimizerAlgorithm = optimizerAlgorithm # 'kalman', 'lm', 'trf', 'dogbox'
        self.kalmanCov = np.eye(len(self.stateCov)) # only used for kalman filter algorithm
        self.curTime = time.time()

    def objectiveFcn(self, x, *args, **kwargs):
        return self.curMeas - self.measFcn(x, *args, **kwargs)

    def estimate(self, meas, action=None, dt=None):
        self.curMeas = meas
        if dt is None:
            t = time.time()
            dt = t - self.curTime
            self.curTime = t
        if self.optimizerAlgorithm == 'kalman': # TODO add bounds
            predictedState = self.stateFcn(state=self.curEstimate, dt=dt, action=action)
            stateJac = self.stateJacobian(x=self.curEstimate, dt=dt, action=action)
            predictedKalmanCov = stateJac.dot(self.kalmanCov).dot(stateJac.T) + self.stateCov
            measJac = self.measJacobian(predictedState)
            invTerm = np.linalg.inv(measJac.dot(predictedKalmanCov).dot(measJac.T) + self.measCov)
            kalmanGain = predictedKalmanCov.dot(measJac.T).dot(invTerm)
            self.curEstimate += kalmanGain.dot(self.curMeas - self.measFcn(predictedState))
            self.kalmanCov = (np.eye(len(self.curEstimate)) - kalmanGain.dot(measJac)).dot(predictedKalmanCov)
        elif self.optimizerAlgorithm == 'lm':
            self.curEstimate = so.least_squares(
                fun=self.objectiveFcn,
                x0=self.curEstimate,
                jac=self.measJacobian,
                method=self.optimizerAlgorithm
            ).x
            self.curEstimate = np.maximum(self.stateBounds[0], np.minimum(self.stateBounds[1], self.curEstimate))
        elif self.optimizerAlgorithm == 'trf' or self.optimizerAlgorithm == 'dogbox':
            self.curEstimate = so.least_squares(
                fun=self.objectiveFcn,
                x0=self.curEstimate,
                jac=self.measJacobian,
                bounds=self.stateBounds,
                method=self.optimizerAlgorithm
            ).x
        return self.curEstimate
