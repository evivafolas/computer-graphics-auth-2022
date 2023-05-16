#
#   Triangle Filling - Computer Graphics AUTh 2022
#       Dimitrios Folas Demiris, AEM: 9415   
#

from cmath import isnan
import numpy as np
import colorInterpolation as clrint
import graphicsUtility

# 
# Smooth Image Rendering
# 

def gouraudRender(verts2d, vcolors, img):
    
    if(verts2d == verts2d[0]).all():
        img[int(verts2d[0,0]), int(verts2d[0,1])] = np.mean(vcolors, axis = 0)

        return img

    edgeVerts , xLimOfEdge, yLimOfEdge, edgeSigma = graphicsUtility.edgeLims(verts2d)

    xMin, xMax = int(np.amin(xLimOfEdge)), int(np.amax(xLimOfEdge))
    yMin, yMax = int(np.amin(yLimOfEdge)), int(np.amax(yLimOfEdge))

    activeEdges = np.array([False, False, False])
    activeNodes = np.zeros((3,2))

    invisible   = False
    nodeEdgeCombo = { 0: [0, 1], 1: [0, 2], 2: [1, 2] }

    activeEdges, activeNodes, invisible = graphicsUtility.activeElems(activeEdges, activeNodes, edgeVerts, yLimOfEdge, edgeSigma)

    if invisible: 
        return img

    for i in range(yMin,yMax):
        
        activeEdges, activeNodes, updatedNodes = graphicsUtility.updateActiveEdge(i, edgeVerts, yLimOfEdge, edgeSigma, activeEdges, activeNodes)
        activeNodes = graphicsUtility.updateActiveNode(edgeSigma, activeEdges, activeNodes, updatedNodes)

        img, activeNodeColor = graphicsUtility.colorcont(i, nodeEdgeCombo, xLimOfEdge, yLimOfEdge, edgeSigma, activeEdges, activeNodes, vcolors, img)

        xLeft, leftID   = np.min(activeNodes[activeEdges, 0]), np.argmin(activeNodes[activeEdges, 0])
        xRight, rightID = np.max(activeNodes[activeEdges, 0]), np.argmax(activeNodes[activeEdges, 0])

        c1, c2 = activeNodeColor[activeEdges][leftID], activeNodeColor[activeEdges][rightID]

        xCount = 0

        for j in range(xMin, xMax +1 ):

            xCount += np.count_nonzero(j == np.around(activeNodes[activeEdges, 0]))

            if xCount % 2 != 0  and int(np.around(xLeft)) != int(np.around(xRight)):
                img[j, i] = clrint.colorInterpolation(int(np.around(xLeft)), int(np.around(xRight)),j , c1, c2)

    return img

# 
# Flat Image Rendering 
# 

def flatRender(verts2d, vcolors, img):
    
    newColor = np.array(np.mean(vcolors, axis=0))

    if(verts2d == verts2d[0]).all():
        img[int(verts2d[0,0]), int(verts2d[0,1])] = newColor

        return img

    edgeVerts , xLimOfEdge, yLimOfEdge, edgeSigma = graphicsUtility.edgeLims(verts2d)

    xMin, xMax = int(np.amin(xLimOfEdge)), int(np.amax(xLimOfEdge))
    yMin, yMax = int(np.amin(yLimOfEdge)), int(np.amax(yLimOfEdge))

    activeEdges = np.array([False, False, False])
    activeNodes = np.zeros((3,2))

    activeEdges, activeNodes, invisible = graphicsUtility.activeElems(activeEdges, activeNodes, edgeVerts, yLimOfEdge, edgeSigma)

    if invisible: 
        return img
    
    for i in range(yMin, yMax + 1):

        xCount = 0 

        for j in range(xMin, xMax + 1):
            xCount += np.count_nonzero(j == np.around(activeNodes[activeEdges][:,0]))

            if xCount % 2!=0:
                
                img[j,i] = newColor

            elif i == yMax and np.count_nonzero(np.around(activeNodes[activeEdges][:,0])) > 0:

                img[j,i] = newColor
        
        activeEdges, activeNodes, updatedNodes = graphicsUtility.updateActiveEdge(i, edgeVerts, yLimOfEdge, edgeSigma, activeEdges, activeNodes)
        activeNodes = graphicsUtility.updateActiveNode(edgeSigma, activeEdges, activeNodes, updatedNodes)

    return img

# 
# Implementation merge
#


def render(verts2d, faces, vcolors, depth, m, n, shade_t):

    img = np.ones((m, n, 3))
    
    depthT = np.array(np.mean(depth[faces], axis=1))
    trianglesOrdered = list(np.flip(np.argsort(depthT)))

    for t in trianglesOrdered:

        triangleVert= faces[t]

        trVerts2d = np.array(verts2d[triangleVert])  
        trVcolors = np.array(vcolors[triangleVert])  

        if shade_t == 'flat':

            img = flatRender(trVerts2d, trVcolors, img)

        elif shade_t == 'gouraud':

            img = gouraudRender(trVerts2d, trVcolors, img)

    return img