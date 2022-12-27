'''
使用gabor filter做分析
'''

import numpy as np
import matplotlib.pyplot as plt
from util import readHandPerformance, cropHandPerformance, handPerformanceToMatrix, \
    scaleToReferenceControlSequence

cropInterval = [1430, 1530]
usedJointAxis = [[0, 'x'], [1, 'x']]

def gabor_1d(x, f, sigma, mu):
    x_rotated = x
    real = np.exp(-np.pi * (x_rotated - mu)**2 / (2 * sigma**2)) * np.cos(2 * np.pi * f * x_rotated)
    imag = np.exp(-np.pi * (x_rotated - mu)**2 / (2 * sigma**2)) * np.sin(2 * np.pi * f * x_rotated)
    return real, imag

def gabor_response_1d(signal, width, f, sigma, mu):
    x = np.arange(width)
    gabor_real, gabor_imag = gabor_1d(x, f, sigma, mu)
    response_real = np.convolve(signal, gabor_real, mode='same')
    response_imag = np.convolve(signal, gabor_imag, mode='same')
    return response_real, response_imag

def apply_gabor_get_freq_phase(signal, freq, sigma, winSize, mu):
    response_phase = np.zeros((len(freq), signal.shape[0]))
    response_amplitude = np.zeros((len(freq), signal.shape[0]))
    for i, _freqsigma in enumerate(zip(freq, sigma)):
        _freq, _sigma = _freqsigma
        _real, _imag = gabor_response_1d(signal, winSize, _freq, _sigma, mu)
        _phase = np.arctan2(_imag, _real)
        _amplitude = np.sqrt(_imag**2 + _real**2)
        response_phase[i, :] = _phase
        response_amplitude[i, :] = _amplitude
    return response_phase, response_amplitude

def main():
    # Read hand performance into a matrix and crop desire interval 
    data = readHandPerformance()
    cropData = cropHandPerformance(data, cropInterval[0], cropInterval[1])
    dataMat = handPerformanceToMatrix(cropData, usedJointAxis)    
    fullDataMat = handPerformanceToMatrix(data, usedJointAxis)

    # Scale data to range of reference control sequence
    # 這個沒有作用, 需要scale的是求出來的phase的數值範圍 
    # fullDataMat[0, :] = scaleToReferenceControlSequence(fullDataMat[0, :], dataMat[0, :])
    # fullDataMat[1, :] = scaleToReferenceControlSequence(fullDataMat[1, :], dataMat[1, :])

    # TODO: Apply gabor to reference control sequence 
    # Extend reference control sequence. Get range of phase. 


    # Apply gabor filter (multiple frequency version) 
    freq = np.reciprocal(np.linspace(5, 150, 50))
    sigma = (2/3) / freq 
    print('freq: ', freq)
    print('sigma: ', sigma)
    response_phase, response_amplitude = apply_gabor_get_freq_phase(
        fullDataMat.T[:, 0], freq, sigma, 150, 0
    )
    response_phase1, response_amplitude1 = apply_gabor_get_freq_phase(
        fullDataMat.T[:, 1], freq, sigma, 150, 0
    )
    # print('max amplitude: ', np.argmax(response_amplitude, axis=0).tolist())
        
    fig, axs = plt.subplots(2)
    fig.suptitle('phase 1')
    axs[0].plot(np.arange(fullDataMat.T[:, 0].shape[0]), fullDataMat.T[:, 0])
    axs[1].plot(response_phase[np.argmax(response_amplitude, axis=0), range(response_phase.shape[1])])
    fig, axs = plt.subplots(2)
    fig.suptitle('phase 2')
    axs[0].plot(np.arange(fullDataMat.T[:, 1].shape[0]), fullDataMat.T[:, 1])
    axs[1].plot(response_phase1[np.argmax(response_amplitude1, axis=0), range(response_phase1.shape[1])])
    plt.show()



if __name__=='__main__':
    main()
    pass