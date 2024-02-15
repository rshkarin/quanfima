import pyvista as pv
from pyvista import examples

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology
from quanfima.utils import geo2rgb
from skimage import measure
import vtk

data = np.memmap('data/polymer3d_8bit_128x128x128.raw',
                 shape=(128,128,128), dtype=np.uint8, mode='r')

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
vol_orient = np.zeros(skel.shape + (3,), dtype=np.float32)

for z, y, x in zip(Z, Y, X):
    vol_orient[z, y, x] = geo2rgb(lat[z, y, x], azth[z, y, x])

depth, height, width = skel.shape

values = vtk.vtkDoubleArray()
values.SetName("values")
# values.SetNumberOfComponents(3)
values.SetNumberOfComponents(1)
# values.SetNumberOfTuples(128*128*128*3)
values.SetNumberOfTuples(128*128*128)
for z in range(depth):
    for y in range(height):
        for x in range(width):
            values.SetValue(z*depth + y*height + x, data[z, y, x])

            # for c in range(3):
            #     values.SetValue(z*depth + y*height + x*width + c, vol_orient[z, y, x, c])

image_data = vtk.vtkImageData()
image_data.SetOrigin(0, 0, 0)
image_data.SetSpacing(1, 1, 1)
image_data.SetDimensions(depth, height, width)
image_data.GetPointData().SetScalars(values)

volumeMapper = vtk.vtkFixedPointVolumeRayCastMapper()
volumeMapper.SetBlendModeToMaximumIntensity()
# volumeMapper = vtk.vtkOpenGLGPUVolumeRayCastMapper()
volumeMapper.SetInputData(image_data)
# volumeMapper.SetBlendModeToMaximumIntensity()

# actor = vtk.vtkImageActor()
# actor.GetMapper().SetInputData(image_data)
# ren = vtk.vtkRenderer()
# renWin = vtk.vtkRenderWindow()
# renWin.AddRenderer(ren)
# renWin.SetWindowName('ReadSTL')
# iren = vtk.vtkRenderWindowInteractor()
# iren.SetRenderWindow(renWin)
# ren.AddActor(actor)
# iren.Initialize()
# renWin.Render()
# iren.Start()
colorFun = vtk.vtkColorTransferFunction()
opacityFun = vtk.vtkPiecewiseFunction()

colorFun.AddRGBSegment(90.0, 1.0, 1.0, 1.0, 255.0, 1.0, 1.0, 1.0)
opacityFun.AddSegment(0.1, 0.0, 255, 1.0)

# colorFunc = vtk.vtkColorTransferFunction()
# colorFunc.AddRGBPoint(10, 1.0, 1.0, 1.0) # Red
# colorFunc.AddRGBPoint(255, 1.0, 1.0, 1.0) # Red
# colorFunc.AddRGBPoint(5, 255, 0.0, 0.0) # Red
# colorFunc.AddRGBPoint(0, 0.0, 200.0, 0.0) # Red
# colorFunc.AddRGBPoint(5, 0.0, 255.0, 0.0) # Red
# colorFunc.AddRGBPoint(0, 0.0, 0.0, 200.0) # Red
# colorFunc.AddRGBPoint(5, 0.0, 0.0, 255.0) # Red

# opacity = vtk.vtkPiecewiseFunction()
# opacity.AddPoint(0, 0.0)
# opacity.AddPoint(255, 1.0)

volumeProperty = vtk.vtkVolumeProperty()
volumeProperty.SetColor(colorFun)
# volumeProperty.SetScalarOpacity(opacityFun)
volumeProperty.SetInterpolationTypeToLinear()
volumeProperty.SetIndependentComponents(3)
volumeProperty.ShadeOff()

volume = vtk.vtkVolume()
volume.SetMapper(volumeMapper)
volume.SetProperty(volumeProperty)

ren = vtk.vtkRenderer()
ren.AddVolume(volume)
ren.SetBackground(125, 125, 125)

renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetSize(900, 900)

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(renWin)

interactor.Initialize()
renWin.Render()
interactor.Start()
