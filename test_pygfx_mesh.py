import numpy as np
import imageio.v3 as iio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import pylinalg as la

from scipy import ndimage as ndi
from skimage import morphology
from quanfima.utils import geo2rgb
from skimage.measure import marching_cubes

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
renderer.blend_mode = "weighted_plus"
scene = gfx.Scene()

scene.add(gfx.Background(None, gfx.BackgroundMaterial("#446")))

radius_structure_elem = 2

azth = np.load("./results/azth.output.npy")
lat = np.load("./results/lat.output.npy")

skel = azth.copy().astype(np.float32)
skel_8bit = azth.copy().astype(np.int32)

skel[skel.nonzero()] = 1.0
skel_8bit[skel_8bit.nonzero()] = 255

azth = ndi.grey_dilation(azth, structure=morphology.ball(radius_structure_elem))
lat = ndi.grey_dilation(lat, structure=morphology.ball(radius_structure_elem))
skel = ndi.binary_dilation(skel, structure=morphology.ball(radius_structure_elem))

skel_8bit = ndi.binary_dilation(skel_8bit, structure=morphology.ball(1))

Z, Y, X = skel.nonzero()
vol_orient = np.zeros(skel.shape + (3,), dtype=np.float32)

for z, y, x in zip(Z, Y, X):
    vol_orient[x, y, z] = geo2rgb(lat[z, y, x], azth[z, y, x])

tex = gfx.Texture(vol_orient, dim=3)

surface = marching_cubes(skel_8bit, 0)

geo = gfx.Geometry(
    positions=surface[0], indices=surface[1], normals=surface[2], texcoords=surface[0]
    # positions=surface[0], indices=surface[1], normals=surface[2],
)

fibers = gfx.Mesh(geo, gfx.MeshPhongMaterial(map=tex))
scene.add(fibers)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.show_object(scene)

controller = gfx.OrbitController(camera)
controller.register_events(renderer)

scene.add(gfx.AmbientLight())
scene.add(gfx.DirectionalLight())
scene.add(gfx.AxesHelper(size=40, thickness=5))

def animate():
    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
