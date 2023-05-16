#
#   Triangle Filling - Computer Graphics AUTh 2022
#       Dimitrios Folas Demiris, AEM: 9415   
#

import numpy as np 

def colorInterpolation(x1,x2,x,c1,c2):

    slope    = (x2 - x) / (x2 - x1)
    colorVal = np.array([slope * c1 + (1 - slope) * c2])

    return colorVal