#
#   Triangle Filling - Computer Graphics AUTh 2022
#       Dimitrios Folas Demiris, AEM: 9415   
#

from cmath import isnan
import numpy as np
import colorInterpolation as color
import matplotlib.pyplot as plt
import imageio
import scipy as sc

# 

def activeElems(activeEdges, activeNodes, edgeVerts, yLimOfEdge, edgeSigma):
    
    yMin, yMax = int(np.amin(yLimOfEdge)), int(np.amax(yLimOfEdge))
    invisible = False

    for i, yLim in enumerate(yLimOfEdge):

        if yLim[0] == yMin:
            
            if edgeSigma[i] == 0:
                
                continue

            if isnan(edgeSigma[i]):

                invisible = True
                continue

            activeEdges[i] = True
            position = np.argmin(edgeVerts[i, :, 1])
            activeNodes[i] = [edgeVerts[i, position, 0], yLimOfEdge[i, 0]]

    return activeEdges, activeNodes, invisible

# 

def edgeLims(verts2d):
    
    edgeVerts = np.array([[verts2d[0], verts2d[1]], [verts2d[0],  verts2d[2]], [verts2d[1], verts2d[2]]])
    
    xLim = np.array([np.min(edgeVerts[:, :, 0], axis=1), np.max(edgeVerts[:, :, 0], axis=1)]).T

    yLim = np.array([np.min(edgeVerts[:, :, 1], axis= 1), np.max(edgeVerts[:, :, 1], axis= 1)]).T

    np.seterr(invalid='ignore')

    delta     = np.array(edgeVerts[:, 1] - edgeVerts[:, 0])
    edgeSigma = np.array(delta[:, 1] / delta[:, 0])

    return edgeVerts, xLim, yLim, edgeSigma
    
# 

def updateActiveEdge(y, edgeVerts, yLimOfEdge, edgeSigma, activeEdges, activeNodes):

    updatedNodes = set()

    for i, yLim in enumerate(yLimOfEdge):
        
        if yLim[0] == y:
            
            if isnan(edgeSigma[i]):
                continue

            activeEdges[i] = True
            position = np.argmin(edgeVerts[i, :, 1])
            activeNodes[i] = [edgeVerts[i, position, 0], yLimOfEdge[i,0]]
            updatedNodes.add(i)
        
        if yLim[1] == y:
            activeEdges[i] = False
    
    return activeEdges, activeNodes, updatedNodes

# 

def updateActiveNode(edgeSigma, activeEdges, activeNodes, updatedNodes):

    for i, sig in enumerate(edgeSigma):

        if activeEdges[i] and sig != 0 and i not in updatedNodes:

            activeNodes[i, 0] += 1 / edgeSigma[i]
            activeNodes[i, 1] += 1

    return activeNodes

# 

def colorcont(y, nodeEdgeCobmo, xLimOfEdge, yLimOfEdge, edgeSigma, activeEdges, activeNodes, vcolors, img):

    activeNodeClr = np.zeros((3,3))

    for i, pt in enumerate(activeNodes):

        if activeEdges[i]:

            xEdge = np.array(xLimOfEdge[i])
            yEdge = np.array(yLimOfEdge[i])

            pairNodes = nodeEdgeCobmo[i]

            c1, c2 = vcolors[pairNodes[0]], vcolors[pairNodes[1]]

            if edgeSigma[i] == 0:

                activeNodeClr[i] = color.colorInterpolation(xEdge[0], xEdge[1], activeNodes[i, 0], c1, c2)

                for j in range(xEdge[0], xEdge[1]):

                    img[int(np.around(j)), int(np.around(y))] = color.colorInterpolation(xEdge[0], xEdge[1], j, c1, c2)

            elif np.abs(edgeSigma[i]) == float('inf'):

                activeNodeClr[i] = color.colorInterpolation(yEdge[0], yEdge[1], y, c1, c2)
                img[int(activeNodeClr[i, 0]), int(np.around(y))] = activeNodeClr[i]

            else:

                activeNodeClr[i] = color.colorInterpolation(yEdge[0], yEdge[1], y, c1, c2)
                img[int(activeNodes[i, 0]), int(np.around(y))] = activeNodeClr[i]

    return img, activeNodeClr                 

# 

def dataLoad(file):

    data = np.load(file, allow_pickle=True).tolist()
    data = dict(data)

    verts2d = np.array(data['verts2d'])
    vcolors = np.array(data['vcolors'])
    faces   = np.array(data['faces']) 
    depth   = np.array(data['depth'])

    return verts2d, vcolors, faces, depth

# 

def displayNumpyImage(img, name):

    plt.imshow(img)
    plt.show()
    imageio.imsave('render/' + name + '.png', (img * 255).astype(np.uint8))
