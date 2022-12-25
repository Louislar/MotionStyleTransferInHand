'''
Construct mapping function for control motion representation
'''

import numpy as np 
import matplotlib.pyplot as plt 
from util import readHandPerformance

def generateComplexSineWave(T):
    '''
    Generate a complex sine wave which is 
    :T: total time
    '''
    phas = np.arange(0, 2*np.pi, 2*np.pi/T)
    complexSineWave = np.exp(1j*phas)
    return np.real(complexSineWave), np.imag(complexSineWave)



def main():
    pass

if __name__=='__main__':
    main()
    real, imag = generateComplexSineWave(100)
    plt.plot(range(len(real)), real, '.-')
    plt.plot(range(len(real)), imag, '.-')
    plt.show()