#
#   Triangle Filling - Computer Graphics AUTh 2022
#       Dimitrios Folas Demiris, AEM: 9415   
#

import graphicsUtility as grutl
import triangleFill

m = 512
n = 512

verts2d, vcolors, faces, depth = grutl.dataLoad(file="hw1.npy")
print("Data loaded.")
img = triangleFill.render(verts2d, faces, vcolors, depth, m, n, shade_t='flat')
print("Image rendered.")
grutl.displayNumpyImage(img,'flatRenderDemo')
print("Image saved.")