from scipy import ndimage as ndi
from skimage import morphology
from quanfima.utils import geo2rgb

from itertools import cycle

import numpy as np

from vispy import app, scene, io
from vispy.color import get_colormaps, BaseColormap
from vispy.visuals.transforms import STTransform

# volume
radius_structure_elem = 1

azth = np.load("./results/azth.output.npy")
lat = np.load("./results/lat.output.npy")

skel = azth.copy().astype(np.float32)
skel[skel.nonzero()] = 1.0

azth = ndi.grey_dilation(azth, structure=morphology.ball(radius_structure_elem))
lat = ndi.grey_dilation(lat, structure=morphology.ball(radius_structure_elem))
skel = ndi.binary_dilation(skel, structure=morphology.ball(radius_structure_elem))

Z, Y, X = skel.nonzero()
vol_orient = np.zeros(skel.shape + (3,), dtype=np.float32)

for z, y, x in zip(Z, Y, X):
    vol_orient[z, y, x] = geo2rgb(lat[z, y, x], azth[z, y, x])

print(vol_orient.shape, vol_orient.size, vol_orient[skel.nonzero()].size)

# Prepare canvas
canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
canvas.measure_fps()

# Set up a viewbox to display the image with interactive pan/zoom
view = canvas.central_widget.add_view()

# Create the volume visuals, only one is visible
volume1 = scene.visuals.Volume(vol_orient, cmap="coolwarm", parent=view.scene, threshold=0.5, texture_format="r32f", plane_position=[0,0,0])
volume1.transform = scene.STTransform(translate=(64, 64, 0))

# Create three cameras (Fly, Turntable and Arcball)
fov = 60.
cam1 = scene.cameras.FlyCamera(parent=view.scene, fov=fov, name='Fly')
cam2 = scene.cameras.TurntableCamera(parent=view.scene, fov=fov,
                                     name='Turntable')
cam3 = scene.cameras.ArcballCamera(parent=view.scene, fov=fov, name='Arcball')
view.camera = cam2  # Select turntable at first

# Create an XYZAxis visual
axis = scene.visuals.XYZAxis(parent=view)
s = STTransform(translate=(50, 50), scale=(50, 50, 50, 1))
affine = s.as_matrix()
axis.transform = affine


# create colormaps that work well for translucent and additive volume rendering
class TransFire(BaseColormap):
    glsl_map = """
    vec4 translucent_fire(float t) {
        return vec4(pow(t, 0.5), t, t*t, max(0, t*1.05 - 0.05));
    }
    """


class TransGrays(BaseColormap):
    glsl_map = """
    vec4 translucent_grays(float t) {
        return vec4(t, t, t, t*0.05);
    }
    """

# Setup colormap iterators
opaque_cmaps = cycle(get_colormaps())
translucent_cmaps = cycle([TransFire(), TransGrays()])
opaque_cmap = next(opaque_cmaps)
translucent_cmap = next(translucent_cmaps)

interp_methods = cycle(volume1.interpolation_methods)
interp = next(interp_methods)


# Implement axis connection with cam2
@canvas.events.mouse_move.connect
def on_mouse_move(event):
    if event.button == 1 and event.is_dragging:
        axis.transform.reset()

        axis.transform.rotate(cam2.roll, (0, 0, 1))
        axis.transform.rotate(cam2.elevation, (1, 0, 0))
        axis.transform.rotate(cam2.azimuth, (0, 1, 0))

        axis.transform.scale((50, 50, 0.001))
        axis.transform.translate((50., 50.))
        axis.update()


# Implement key presses
@canvas.events.key_press.connect
def on_key_press(event):
    global opaque_cmap, translucent_cmap
    if event.text == '1':
        cam_toggle = {cam1: cam2, cam2: cam3, cam3: cam1}
        view.camera = cam_toggle.get(view.camera, cam2)
        print(view.camera.name + ' camera')
        if view.camera is cam2:
            axis.visible = True
        else:
            axis.visible = False
    elif event.text == '2':
        methods = ['mip', 'translucent', 'iso', 'additive']
        method = methods[(methods.index(volume1.method) + 1) % 4]
        print("Volume render method: %s" % method)
        cmap = opaque_cmap if method in ['mip', 'iso'] else translucent_cmap
        volume1.method = method
        volume1.cmap = cmap
    elif event.text == '3':
        volume1.visible = not volume1.visible
    elif event.text == '4':
        if volume1.method in ['mip', 'iso']:
            cmap = opaque_cmap = next(opaque_cmaps)
        else:
            cmap = translucent_cmap = next(translucent_cmaps)
        volume1.cmap = cmap
    elif event.text == '5':
        interp = next(interp_methods)
        volume1.interpolation = interp
        print(f"Interpolation method: {interp}")
    elif event.text == '0':
        cam1.set_range()
        cam3.set_range()
    elif event.text != '' and event.text in '[]':
        s = -0.025 if event.text == '[' else 0.025
        volume1.threshold += s
        th = volume1.threshold
        print("Isosurface threshold: %0.3f" % th)

if __name__ == '__main__':
    print(__doc__)
    app.run()
