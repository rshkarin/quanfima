import pyvista as pv
from pyvista import examples

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology
from quanfima.utils import geo2rgb
from skimage import measure
import vtk

# points = np.array(
#     [
#         [0.0480, 0.0349, 0.9982],
#         [0.0305, 0.0411, 0.9987],
#         [0.0207, 0.0329, 0.9992],
#         [0.0218, 0.0158, 0.9996],
#         [0.0377, 0.0095, 0.9992],
#         [0.0485, 0.0163, 0.9987],
#         [0.0572, 0.0603, 0.9965],
#         [0.0390, 0.0666, 0.9970],
#         [0.0289, 0.0576, 0.9979],
#         [0.0582, 0.0423, 0.9974],
#         [0.0661, 0.0859, 0.9941],
#         [0.0476, 0.0922, 0.9946],
#         [0.0372, 0.0827, 0.9959],
#         [0.0674, 0.0683, 0.9954],
#     ],
# )
#
#
# face_a = [6, 0, 1, 2, 3, 4, 5]
# face_b = [6, 6, 7, 8, 1, 0, 9]
# face_c = [6, 10, 11, 12, 7, 6, 13]
# faces = np.concatenate((face_a, face_b, face_c))
#
# colors = []
# for _ in range(3):
#     colors.append([255, 0, 0])
#
# mesh = pv.PolyData(points, faces)
# mesh.cell_data['colors'] = colors
#
# plotter = pv.Plotter()
# _ = plotter.add_mesh(
#     mesh,
#     scalars='colors',
#     # lighting=False,
#     rgb=True,
#     preference='cell',
# )
# plotter.show()


radius_structure_elem = 1

azth = np.load("./results/azth.output.npy")
lat = np.load("./results/lat.output.npy")

common_vol = np.abs(azth) + np.abs(lat)

skel = common_vol.copy().astype(np.float32)
skel[skel.nonzero()] = 255.0

azth = ndi.grey_dilation(azth, structure=morphology.ball(radius_structure_elem + 1))
lat = ndi.grey_dilation(lat, structure=morphology.ball(radius_structure_elem + 1))
skel2 = ndi.binary_dilation(skel, structure=morphology.ball(radius_structure_elem))

Z, Y, X = skel2.nonzero()
vol_orient = np.zeros(skel2.shape + (3,), dtype=np.float32)

for z, y, x in zip(Z, Y, X):
    vol_orient[z, y, x] = geo2rgb(lat[z, y, x], azth[z, y, x])

# verts_azth, faces_azth, normals_azth, values_azth = measure.marching_cubes(azth, azth.min())
# vfaces_azth = np.column_stack((np.ones(len(faces_azth),) * 3, faces_azth)).astype(int)
# print(verts_azth[0], faces_azth[0], normals_azth[0], len(values_azth))
#
# verts_lat, faces_lat, normals_lat, values_lat = measure.marching_cubes(lat, lat.min())
# vfaces_lat = np.column_stack((np.ones(len(faces_lat),) * 3, faces_lat)).astype(int)
# print(verts_lat[0], faces_lat[0], normals_lat[0], len(values_lat))

verts, faces, normals, values = measure.marching_cubes(skel2, 0)
vfaces = np.column_stack((np.ones(len(faces),) * 3, faces)).astype(int)
print(verts[0], faces[0], normals[0], len(values), len(verts), len(faces))

mesh_common = pv.PolyData(verts, vfaces)
mesh_common['Normals'] = normals
mesh_common['values'] = values

colors = []
for p1, p2, p3 in faces:
    coords_p1, coords_p2, coords_p3 = verts[p1], verts[p2], verts[p3]
    color = (np.array(vol_orient[round(coords_p1[2]), round(coords_p1[1]), round(coords_p1[0])]) + np.array(vol_orient[round(coords_p2[2]), round(coords_p2[1]), round(coords_p2[0])]) + np.array(vol_orient[round(coords_p3[2]), round(coords_p3[1]), round(coords_p3[0])])) / 3.
    # color = (np.array(vol_orient[round(coords_p1[0]), round(coords_p1[1]), round(coords_p1[2])]) + np.array(vol_orient[round(coords_p2[0]), round(coords_p2[1]), round(coords_p2[2])]) + np.array(vol_orient[round(coords_p3[0]), round(coords_p3[1]), round(coords_p3[2])])) / 3.
    colors.append(color)

mesh_common.cell_data['colors'] = colors 
# mesh_azth = pv.PolyData(verts_azth, vfaces_azth)
# mesh_azth['Normals'] = normals_azth
# mesh_azth['values'] = values_azth
#
# mesh_lat = pv.PolyData(verts_lat, vfaces_lat)
# mesh_lat['Normals'] = normals_lat
# mesh_lat['values'] = values_lat
#
# merged_mesh = mesh_azth.merge(mesh_lat)

smoothed_mesh = mesh_common.smooth_taubin()

plotter = pv.Plotter()
_ = plotter.add_mesh(
    smoothed_mesh,
    scalars='colors',
    lighting=False,
    rgb=True,
    preference='cell',
)
plotter.show()

# smoothed_mesh = merged_mesh.smooth_taubin()
#
# smoothed_mesh.plot(scalars='values')
# mesh_azth.plot(scalars='values')
# mesh_lat.plot(scalars='values')

# print(len(vol_orient.flatten(order="F")), vol_orient.flatten(order="F")[0])

# side_size = skel_8bit.shape[0]
# xi = np.arange(side_size)
# z, y, x, d = np.meshgrid(xi, xi, xi, 3)

# azth_vol = pv.ImageData(dimensions=(128, 128, 128))
# lat_vol = pv.ImageData(dimensions=(128, 128, 128))
#
# azth_vol.point_data["values"] = azth.flatten(order="F")
# lat_vol.point_data["values"] = lat.flatten(order="F")
#
# lat_vol = lat_vol.gaussian_smooth(std_dev=1.0)
# azth_vol = azth_vol.gaussian_smooth(std_dev=1.0)
# smoothed_data = data.gaussian_smooth(std_dev=3.0)
#
# dargs = dict(clim=lat_vol.get_data_range(), opacity=[0, 0, 0, 0.1, 0.3, 0.6, 1])

# n = [100, 150, 200, 245, 255]

# p = pv.Plotter()

# _ = p.add_mesh(
#     mesh,
    # scalars='colors',
    # lighting=False,
    # rgb=True,
    # preference='cell',
# )
# p.subplot(0, 0)
# p.add_text("Original Image", font_size=24)
# p.add_mesh(data.contour(n), **dargs)
# p.add_volume(azth_vol, **dargs)
# p.add_volume(vol_orient, **dargs)
# p.subplot(0, 1)
# p.add_text("Gaussian smoothing", font_size=24)
# p.add_mesh(smoothed_data.contour(n), **dargs)
# p.add_volume(smoothed_data, **dargs)
# p.link_views()
# p.camera_position = [(-162.0, 704.8, 65.02), (90.0, 108.0, 90.0), (0.0068, 0.0447, 0.999)]
# p.show()
