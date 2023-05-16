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
    
    if (verts2d == verts2d[0]).all():
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

        for j in range(xMin, xMax + 1):

            xCount = xCount + np.count_nonzero(j == np.around(activeNodes[activeEdges, 0]))

            if xCount % 2 != 0 :

                if j < img.shape[0] and j >= 0 and i < img.shape[1] and i >= 0: 

                    img[j, i] = clrint.colorInterpolation(int(np.around(xLeft)), int(np.around(xRight)),j , c1, c2)

    return img



# 
# Implementation merge
#


def render(verts2d, faces, vcolors, depth, imh, imw, shade_t):

    assert imh > 0 and imw > 0 

    img = np.ones((imh, imw, 3))
    
    depthT = np.array(np.mean(depth[faces], axis=1))
    
    trianglesOrdered = list(np.flip(np.argsort(depthT)))

    for t in trianglesOrdered:

        triangleVert= faces[t]

        trVerts2d = np.array(verts2d[triangleVert])  
        trVcolors = np.array(vcolors[triangleVert])  

        # if shade_t == 'flat':
            # img = flatRender(trVerts2d, trVcolors, img)

        if shade_t == 'gouraud':

            img = gouraudRender(trVerts2d, trVcolors, img)

    return img