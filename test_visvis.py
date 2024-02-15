import numpy as np
import visvis as vv
import numpy as np

from scipy import ndimage as ndi
from skimage import morphology
from quanfima.utils import geo2rgb

app = vv.use('pyside6')

radius_structure_elem = 1

# azth = np.load("./results/azth.output.npy")
# lat = np.load("./results/lat.output.npy")

azth = np.memmap('./results/azth.raw',
                 shape=(128,128,128), dtype=np.float32, mode='r')

lat = np.memmap('./results/lat.raw',
                 shape=(128,128,128), dtype=np.float32, mode='r')

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

vv.figure()
t2 = vv.volshow(vol_orient[8:-8,8:-8,8:-8], renderStyle = 'iso')
t2.isoThreshold = 0.5
vv.title('color ISO-surface render')

# Run app
app.Run()
