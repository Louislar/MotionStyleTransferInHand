'''
重現Motion editing with data glove的system
'''

def main():
    # 1.1 read hand rotation data
    # 1.2 read body rotation data
    # 2. Average filter apply to both rotation curves
    # 3. Low pass filter apply to both rotation curves via FFT
    # 4. Compute tangent in both curves and find the time point that tanget is 0
    # 5. Construct multiple mapping functions with discrete sample points
    # 5.1 Index finger to right shoulder
    # 5.2 Index finger to left upper leg
    # 5.3 Index finger to left knee
    # 5.4 Middle finger to left shoulder
    # 5.5 Middle finger to right shoulder
    # 5.6 Middle finger to right shoulder
    # 6. Use B-Spline fitting to interpolate each mapping function
    # 7. Store each mapping function in file

    # 1. 
    handRotDirPath = ''
    pass

if __name__=='__main__':
    main()
    pass