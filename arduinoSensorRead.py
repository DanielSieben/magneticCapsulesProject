import serial
import numpy as np
from pathos.multiprocessing import Pool
from functools import partial
import warnings


def getSingleValue(idx, rawData, dataLen, conversionFactor):
    return float(rawData[idx * dataLen: (idx + 1) * dataLen]) * conversionFactor


class arduinoICMSensorArray:
    def __init__(self, comPort: str, baudRate: int, timeout: float = 5., dataLen: int = 10, conversionFactor: float = 1.):
        # inputs:
        #   comPort = communication port for arduino (see device manager ports), typically 'COM#', string
        #   baudRate = sensor baud rate, typically 115200, int
        #   timeout = read error timeout length in seconds, float
        self.ser = serial.Serial(comPort, baudRate, timeout=timeout)
        self.dataLen = dataLen # data length from serial.readline(), including spaces (e.g. ' -00252.90' = 10)
        self.readDim, self.readLen = self.getReadDims()
        self.curRead = np.full(self.readDim, np.nan)
        self.conversionFactor = conversionFactor # conversion factor for data
        self.offsets = np.zeros(self.curRead.shape)
        self.covariance = np.eye(len(self.curRead)) * 1e-8
        self.pool = Pool() # for multiprocessing

    def __del__(self):
        self.ser.close()

    def getReadDims(self):
        rawData = self.ser.readline().decode()
        return int(len(' ' + rawData.replace(' \r\n', '')) / self.dataLen), len(rawData)

    def read(self):
        try:
            rawData = ' ' + self.ser.read(self.readLen).decode()
            for idx in range(self.readDim):
                self.curRead[idx] = getSingleValue(idx, rawData, self.dataLen, self.conversionFactor)
            return self.curRead
        except ValueError or TypeError:
            self.readFailed()
            return self.curRead

    def mp_read(self): # multiprocessing implementation, faster with larger readDim
        try:
            rawData = ' ' + self.ser.read(self.readLen).decode()
            fcn = partial(getSingleValue, rawData=rawData, dataLen=self.dataLen, conversionFactor=self.conversionFactor)
            self.curRead[:] = self.pool.map(fcn, list(range(self.readDim)))
            return self.curRead
        except ValueError or TypeError:
            self.readFailed()
            return self.curRead

    def calibrateSensors(self, nSamples: int = 100):
        samples = np.zeros((nSamples, len(self.curRead)))
        for idx in range(nSamples):
            print('Calibrating sample %d of %d' % (idx + 1, nSamples))
            samples[idx, :] = self.read()
        self.offsets = np.mean(samples, axis=0)
        self.covariance = np.diag(np.var(samples, axis=0))
        print('Calibration complete.')

    def readFailed(self):
        warnings.warn('Serial Read Unsuccessful.')
        self.curRead[:] = np.nan
