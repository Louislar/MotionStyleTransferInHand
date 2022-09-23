import matplotlib.pyplot as plt 
from pymo.parsers import BVHParser
from pymo.viz_tools import *
from pymo.preprocessing import *

def extractDataFromMocapData():
    '''
    TODO: 
    MocapData is a data structure defined in the pymo library. 
    I only want to extract the XYZ positions from some joints in the MocapData. 
    Easy access points is the MocapData.values is a pd.DataFrame. 

    The desire joints: 
    1. left/right feet
    '''
    pass

parser = BVHParser()

parsed_data = parser.parse('data/swimming/125/125_01.bvh')

print_skel(parsed_data)

mp = MocapParameterizer('position')

positions = mp.fit_transform([parsed_data])

print(type(positions))
print(type(positions[0].values))
print(positions[0].values.shape)
print(positions[0].values.columns)

draw_stickfigure(positions[0], frame=10)

# plt.show()