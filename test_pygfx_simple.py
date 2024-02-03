import pygfx as gfx
import pylinalg as la
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from skimage import morphology
from quanfima.utils import geo2rgb
from skimage.measure import marching_cubes

radius_structure_elem = 1

azth = np.load("./results/azth.output.npy")
lat = np.load("./results/lat.output.npy")

skel = azth.copy().astype(np.float32)
skel_8bit = azth.copy().astype(np.int32)

skel[skel.nonzero()] = 1.0

azth = ndi.grey_dilation(azth, structure=morphology.ball(radius_structure_elem))
lat = ndi.grey_dilation(lat, structure=morphology.ball(radius_structure_elem))
skel = ndi.binary_dilation(skel, structure=morphology.ball(radius_structure_elem))

Z, Y, X = skel.nonzero()
vol_orient = np.zeros(skel.shape + (4,), dtype=np.float32)

for z, y, x in zip(Z, Y, X):
    vol_orient[z, y, x] = np.array(list(geo2rgb(lat[z, y, x], azth[z, y, x])) + [1.0])

tex = gfx.Texture(vol_orient, dim=4)
material = gfx.VolumeBasicMaterial(clim=(0, 1), map=tex)

geometry = gfx.Geometry(grid=skel)

vol = gfx.Volume(geometry, material)

if __name__ == "__main__":
    gfx.show(vol)
