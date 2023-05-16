#   Dimitrios Folas Demiris, AEM: 9415   
#       Computer Graphics, ECE AUTh 2022
 
from fnmatch import translate

from matplotlib import image
import graphicsUtility
import sys

imh = 512
imw = 512
camh = 15
camw = 15

f = 70

#   Data Load

vcolors, faces, verts3d, u, ck, cu, cv, t1, t2, phi = graphicsUtility.dataLoad3d(file='hw2.npy')

print('Data loaded.')
image1 = graphicsUtility.renderObject(verts3d, faces, vcolors, imh, imw, camh, camw, f, cv, ck, cu)

print('Image Rendered.')
graphicsUtility.displayNumpyImage(image1, name='imageOG')
print('Image Displayed & Saved.')

#   Transformation 1

transformation = graphicsUtility.transformationMatrix()

transformation.translate(t=t1)
verts3d = graphicsUtility.affineTransform(verts3d, transformation.T)

image2 = graphicsUtility.renderObject(verts3d, faces, vcolors, imh, imw, camh, camw, f, cv, ck, cu)
graphicsUtility.displayNumpyImage(image2, 'transformation1')

#   Transformation 2

transformation = graphicsUtility.transformationMatrix()

transformation.rotate(phi, u)
verts3d = graphicsUtility.affineTransform(verts3d, transformation.T)

image3 = graphicsUtility.renderObject(verts3d, faces, vcolors, imh, imw, camh, camw, f, cv, ck, cu)
graphicsUtility.displayNumpyImage(image3, 'transformation2')

#   Transformation 3

transformation = graphicsUtility.transformationMatrix()

transformation.translate(t=t2)
verts3d = graphicsUtility.affineTransform(verts3d, transformation.T)

image4 = graphicsUtility.renderObject(verts3d, faces, vcolors, imh, imw, camh, camw, f, cv, ck, cu)
graphicsUtility.displayNumpyImage(image4, 'transformation3')

print('All Transformations and Photo Renders Complete.')