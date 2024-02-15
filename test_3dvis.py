import numpy as np
from scipy import ndimage as ndi
from skimage import morphology
from wgpu.gui.auto import WgpuCanvas, run
from quanfima.utils import geo2rgb
import pygfx as gfx
from skimage.measure import marching_cubes


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

radius_structure_elem = 1

azth = np.load("./results/azth.output.npy")
lat = np.load("./results/lat.output.npy")

skel = azth.copy().astype(np.float32)
skel_8bit = azth.copy().astype(np.int32)

skel[skel.nonzero()] = 1.0
skel_8bit[skel_8bit.nonzero()] = 255

azth = ndi.grey_dilation(azth, structure=morphology.ball(radius_structure_elem))
lat = ndi.grey_dilation(lat, structure=morphology.ball(radius_structure_elem))
skel = ndi.binary_dilation(skel, structure=morphology.ball(radius_structure_elem))

Z, Y, X = skel.nonzero()
vol_orient = np.zeros(skel.shape + (3,), dtype=np.float32)

print(vol_orient.size, vol_orient[skel.nonzero()].size)

for z, y, x in zip(Z, Y, X):
    vol_orient[z, y, x] = geo2rgb(lat[z, y, x], azth[z, y, x])

tex = gfx.Texture(vol_orient, dim=3)

surface = marching_cubes(skel_8bit, 0)

print(f"surface = {surface[0].shape}")
# print(f"faces = {faces[-1]}")
# print(f"normals = {normals[-1]}")
# print(f"values = {values[-1]}")
geo = gfx.Geometry(
    positions=surface[0], indices=surface[1], normals=surface[2]
)
geo = gfx.torus_knot_geometry()
mesh = gfx.Mesh(
    geo,
    gfx.MeshPhongMaterial(),
)

torus = gfx.Mesh(gfx.torus_knot_geometry(100, 20, 128, 32), gfx.MeshPhongMaterial())
torus.local.x -= 150
scene.add(torus)

scene.add(mesh)


# voldata = np.memmap("./data/polymer3d_8bit_128x128x128.raw", dtype=np.uint8, shape=(128, 128, 128))
# tex = gfx.Texture(voldata, dim=3)

geometry = gfx.Geometry(grid=tex)
# geometry.colors = vol_orient.reshape((128*128*128, 4))
# gfx.VolumeRayMaterial(clim=(0, 3.14), map=gfx.cm.cividis),
# mat = gfx.VolumeRayMaterial(clim=(0, 1)) #gfx.VolumeBasicMaterial(clim=(0, 3.14), map=tex)
mat = gfx.VolumeMipMaterial(
    clim=(0, 1.0)
)  # gfx.VolumeBasicMaterial(clim=(0, 3.14), map=tex)
vol = gfx.Volume(geometry, mat)
# scene.add(vol)

# slice = gfx.Volume(
#     gfx.Geometry(grid=tex),
#     gfx.VolumeSliceMaterial(plane=(0, 0, 1, 0), clim=(0, 2000)),
# )
# scene.add(vol, slice)

# for ob in (slice, vol):
#     ob.local.position = [-0.5 * i for i in voldata.shape[::-1]]

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.show_object(scene, view_dir=(-1, -1, -1), up=(0, 0, 1))
controller = gfx.OrbitController(camera, register_events=renderer)


@vol.add_event_handler("pointer_down")
def handle_event(event):
    if "Shift" not in event.modifiers:
        return
    info = event.pick_info
    if "index" in info:
        x, y, z = (max(1, int(i)) for i in info["index"])
        print("Picking", x, y, z)
        tex.data[z - 1 : z + 1, y - 1 : y + 1, x - 1 : x + 1] = 2000
        tex.update_range((x - 1, y - 1, z - 1), (3, 3, 3))


def animate():
    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
