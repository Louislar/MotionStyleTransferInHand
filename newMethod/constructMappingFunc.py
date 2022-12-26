'''
Construct mapping function for control motion representation
'''

import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.linear_model import Ridge
from util import readHandPerformance, cropHandPerformance, handPerformanceToMatrix

cropInterval = [1430, 1530]
usedJointAxis = [[0, 'x'], [1, 'x']]

def generateComplexSineWave(T):
    '''
    Generate a complex sine wave which is 
    :T: total time
    '''
    phas = np.arange(0, 2*np.pi, 2*np.pi/T)
    complexSineWave = np.exp(1j*phas)
    return np.real(complexSineWave), np.imag(complexSineWave)

def generateSingleZ(T):
    '''
    Generate z sequence for non cyclic lienar motion transform
    '''
    return np.linspace(0, 1, T)


def computeGaussianProcessRegression(X, Y):
    kernel = ConstantKernel(1.0, (1e-1, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e2))
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=2e-2)
    gaussian_process.fit(X, Y)
    return gaussian_process

def constructRidgeRegression(X, Y):
    ridgeModel = Ridge(alpha=1e-3)
    ridgeModel.fit(X, Y)
    return ridgeModel

def fitDataToComplexSineWave(data, modelConstructFunc): 
    '''
    利用gaussian process regression求data與complex sine wave之間的mapping 
    :data: np.array with dimension numberOfDataPoint * numberOfFeature 
    Output
    :model: a gpr model implies a linear mapping from data to complex sine wave 
    '''
    dataTimeCount = data.shape[0]
    # Generate complex sine wave with number of time points same as the data 
    realWave, imagWave = generateComplexSineWave(dataTimeCount)
    complexWave = np.concatenate([realWave[:, np.newaxis], imagWave[:, np.newaxis]], axis=1)
    
    # fit gpr model
    model = modelConstructFunc(data, complexWave)
    return model

def fitDataToSingleZ(data, modelConstructFunc):
    dataTimeCount = data.shape[0]
    # generate single z 
    singleZ = generateSingleZ(dataTimeCount)

    # fit gpr model 
    model = modelConstructFunc(data, singleZ)
    return model

def main():
    # Read hand performance into a matrix and crop desire interval 
    data = readHandPerformance()
    cropData = cropHandPerformance(data, cropInterval[0], cropInterval[1])
    dataMat = handPerformanceToMatrix(cropData, usedJointAxis)

    # Fit a linear transform for complex sine wave 
    mappingModel = fitDataToComplexSineWave(dataMat.T, computeGaussianProcessRegression)
    # mappingModel = fitDataToSingleZ(dataMat.T, computeGaussianProcessRegression)
    # mappingModel = fitDataToSingleZ(dataMat.T, constructRidgeRegression)

    # Test linear transform model 
    fullDataMat = handPerformanceToMatrix(data, usedJointAxis)
    result = mappingModel.predict(fullDataMat.T)
    print(result.shape)
    plt.plot(range(len(result[:, 1])), result[:, 0], '.-')
    plt.plot(range(len(result[:, 1])), result[:, 1], '.-')
    # plt.plot(range(len(result)), result, '.-')
    plt.show()


    pass

if __name__=='__main__':
    main()
    # real, imag = generateComplexSineWave(100)
    # plt.plot(range(len(real)), real, '.-')
    # plt.plot(range(len(real)), imag, '.-')
    # plt.show()