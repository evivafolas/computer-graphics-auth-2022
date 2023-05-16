#
#   Triangle Filling - Computer Graphics AUTh 2022
#       Dimitrios Folas Demiris, AEM: 9415   
#

from cmath import isnan
from matplotlib import image
import numpy as np
import colorInterpolation as color
import matplotlib.pyplot as plt
import imageio
import scipy as sc
from triangleFill import render

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
    
    xLim = np.array([np.min(edgeVerts[:, :, 0], axis=1), 
                    np.max(edgeVerts[:, :, 0], axis=1)]).T

    yLim = np.array([np.min(edgeVerts[:, :, 1], axis= 1),
                    np.max(edgeVerts[:, :, 1], axis= 1)]).T

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

            elif np.abs(edgeSigma[i]) == float('inf'):

                activeNodeClr[i] = color.colorInterpolation(yEdge[0], yEdge[1], y, c1, c2)

            else:

                activeNodeClr[i] = color.colorInterpolation(yEdge[0], yEdge[1], y, c1, c2)
                
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

def dataLoad3d(file):

    data = np.load(file, allow_pickle=True).tolist()
    data = dict(data)
    
    verts3d = np.array(data['verts3d'])
    vcolors = np.array(data['vcolors'])
    faces = np.array(data['faces'])
    
    u = data['u']
    ck = data['c_lookat']
    cu = data['c_up']
    cv = data['c_org']
    t1 = data['t_1']
    t2 = data['t_2']
    phi = data['phi']

    return vcolors, faces, verts3d, u, ck, cu, cv, t1, t2, phi


# 
 
def displayNumpyImage(img, name):

    # plt.imshow(img)
    # plt.show()

    imageio.imsave('render/' + name + '.png', (img * 255).astype(np.uint8))

#

def renderObject(vert, faces, vcolors, imh, imw, camh, camw, f, cv, cLookat, cUp):

    P, D = projectCameraLookat(f, cv, cLookat, cUp, vert)

    vertRasterize = rasterize(P, imh, imw, camh, camw)
    # print(vertRasterize)
    img = render(verts2d=vertRasterize, faces=faces, vcolors=vcolors, depth=D, imh=imh, imw=imw, shade_t='gouraud')

    return img

#

def rasterize(vert, imh, imw, camh, camw):

    numPoints = vert.shape[0]

    vertRasterizeTemp = np.zeros((numPoints, 2))

    vertical = imh / camh
    horizontal = imw / camw

    for i in range(numPoints):
    
        vertRasterizeTemp[i, 0] = np.around((vert[i, 0] + camh / 2) * vertical - 0.5)
        vertRasterizeTemp[i, 1] = np.around((vert[i, 1] + camw / 2) * horizontal - 0.5)

    vertRasterize = np.zeros((vertRasterizeTemp.shape[0], 2))

    vertRasterize[:, 0] = vertRasterizeTemp[:, 1]
    vertRasterize[:, 1] = vertRasterizeTemp[:, 0] 

    return vertRasterize

#

def affineTransform(cp, T):

    cpAugmented = np.concatenate((cp.T, np.ones((1, cp.shape[0]))), axis=0)
    cqAugmented = np.matmul(T, cpAugmented)

    return cqAugmented.T[:, 0:3]

#

def systemTransform(cp, T):

    cpAugmented = np.concatenate((cp.T, np.ones((1, cp.shape[0]))), axis=0)
    dpAugmented = np.matmul(np.linalg.inv(T), cpAugmented)

    return dpAugmented.T[:, 0:len(dpAugmented) - 1]

#

class transformationMatrix:

    def __init__(self):
        self.T = np.diag(np.ones(4, ))

    def rotate(self, angle, rotAxis):

        u = rotAxis / np.linalg.norm(rotAxis)
        theta = angle

        m1 = np.array([[u[0] ** 2, u[0] * u[1], u[0] * u[2]],
            [u[0] * u[1], u[1] ** 2, u[1] * u[2]],
            [u[0] * u[2], u[1] * u[2], u[2] ** 2]])
        
        m2 = np.diag(np.ones(3, ))
        m3 = np.array([[0, -u[2], u[1]],
                       [u[2], 0, -u[0]],
                       [-u[1], u[0], 0]])

        rotMatrix = (1 - np.around(np.cos(theta))) * m1 + np.around(np.cos(theta)) * m2 + np.around(np.sin(theta)) * m3
        
        self.T[0:len(self.T) - 1, 0:len(self.T) - 1] = rotMatrix

    def translate(self, t): 
        
        self.T[0:len(self.T) - 1, len(self.T) - 1] = t

#

def project_camera(f, cv, cx, cy, cz, p):

    rotMatrix = np.array([cx, cy, cz]).T
    
    m = transformationMatrix()

    m.T[0:len(m.T) - 1, 0:len(m.T) - 1] = rotMatrix
    m.T[0:len(m.T) - 1, len(m.T) - 1] = cv

    m.T = np.around(m.T, 4)

    P = np.zeros((p.shape[0], 2))
    D = np.zeros((p.shape[0], ))

    for i in range(p.shape[0]):
        
        cp = systemTransform(np.array([p[i,:]]), m.T)

        cp = np.concatenate((cp, np.array([np.ones((1,))])), axis = 1)
        
        P[i, 0] = - (f * cp[0, 0] / cp[0, 2])
        P[i, 1] = - (f * cp[0, 1] / cp[0, 2])
        D[i] = cp[0, 2]

    return P, D

#

def projectCameraLookat(f, cv, cLookat, cUp, p):

    zc = np.around((cLookat - cv) / np.linalg.norm(cLookat - cv), 4)
    
    t = cUp - np.dot(cUp, zc) * zc

    yc = np.around(t / np.linalg.norm(t), 4)
    xc = np.around(np.cross(yc, zc), 4)

    return project_camera(f, cv, xc, yc, zc, p)