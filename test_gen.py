import numpy as np
import os
from skimage import filters
from quanfima import morphology as mrph
from quanfima import structure_tensor_3d as tensor3d
from quanfima import raycasting_cpu as raycast_cpu
from quanfima import diameter
from quanfima import visualization as vis
from quanfima import utils
from PIL import Image
from quanfima import simulation

volume, lat_ref, azth_ref, diameter, n_generated, elapsed_time = simulation.simulate_fibers((512,512,512), n_fibers=1, radius_lim=(8, 8), lat_lim=(np.deg2rad(45), np.deg2rad(45)), azth_lim=(np.deg2rad(30), np.deg2rad(30)))

lat_ref.astype(np.float32).tofile("./results/ref_lat_512x512x512_32bit.raw")
azth_ref.astype(np.float32).tofile("./results/ref_azth_512x512x512_32bit.raw")
volume.astype(np.int8).tofile("./results/ref_volume_512x512x512_8bit.raw")
diameter.astype(np.float32).tofile("./results/ref_diam_512x512x512_32bit.raw")

pdata, pskel, pskel_thick = utils.prepare_data(volume)
lat, azth = tensor3d.estimate_3d_orientation(pskel, pdata, 32)

lat.tofile("./results/lat_512x512x512_32bit.raw")
azth.tofile("./results/azth_512x512x512_32bit.raw")

vis.plot_3d_orientation_map('polymer_w32', lat_ref, azth_ref,
                            radius_structure_elem=0,
                            output_dir='./results',
                            camera_azth=40.47,
                            camera_elev=32.5,
                            camera_fov=35.0,
                            camera_loc=(40.85, 46.32, 28.85),
                            camera_zoom=0.005124)

vis.plot_3d_orientation_map('polymer_w32', lat, azth,
                            radius_structure_elem=0,
                            output_dir='./results',
                            camera_azth=40.47,
                            camera_elev=32.5,
                            camera_fov=35.0,
                            camera_loc=(40.85, 46.32, 28.85),
                            camera_zoom=0.005124)

# # vis.plot_3d_diameter_map('polymer_w32', diam,
# #                          output_dir='./results',
# #                          measure_quantity='vox',
# #                          camera_azth=40.47,
# #                          camera_elev=32.5,
# #                          camera_fov=35.0,
# #                          camera_loc=(40.85, 46.32, 28.85),
# #                          camera_zoom=0.005124,
# #                          cb_x_offset=5,
# #                          width=620)
#
#
#
#
# data = np.memmap('data/polymer3d_8bit_128x128x128.raw',
#                  shape=(128,128,128), dtype=np.uint8, mode='r')
#
# data_seg = np.zeros_like(data, dtype=np.uint8)
# for i in range(data_seg.shape[0]):
#   th_val = filters.threshold_otsu(data[i])
#   data_seg[i] = (data[i] > th_val).astype(np.uint8)
#
# data_seg.tofile('./results/data_seg.raw')
#
# # estimate porosity
# pr = mrph.calc_porosity(data_seg)
# for k,v in pr.items():
#   print(f'Porosity ({k}): {v}')
#
# lat_path, azth_path = "./results/lat_ray.output.npy", "./results/azth_ray.output.npy" 
#
# lat, azth = None, None
#
# if True: #not os.path.exists(lat_path) and not os.path.exists(azth_path):  
#     # prepare data and analyze fibers
#     pdata, pskel, pskel_thick = utils.prepare_data(data_seg)
#
#     pskel.tofile("./results/pskel.raw")
#     pdata.tofile("./results/pdata.raw")
#     pskel_thick.tofile("./results/pskel_thick.raw")
#
#     lat, azth = tensor3d.estimate_3d_orientation(pskel, pdata, 16)
#     # lat, azth = raycast_cpu.estimate_3d_orientation(pdata, pskel)
#
#     # diam = diameter.estimate_3d_fiber_diameter_gpu(pskel, pdata, lat, azth, 180)
#
#     np.save(lat_path, lat)
#     np.save(azth_path, azth)
#
#     lat.tofile("./results/lat_ray.raw")
#     azth.tofile("./results/azth_ray.raw")
#
# if lat is None and azth is None:
#     print("opening")
#     lat = np.load(lat_path)
#     azth = np.load(azth_path)
#
# # plot results
# vis.plot_3d_orientation_map('polymer_w32', lat, azth,
#                             output_dir='./results',
#                             camera_azth=40.47,
#                             camera_elev=32.5,
#                             camera_fov=35.0,
#                             camera_loc=(40.85, 46.32, 28.85),
#                             camera_zoom=0.005124)
#
# # vis.plot_3d_diameter_map('polymer_w32', diam,
# #                          output_dir='./results',
# #                          measure_quantity='vox',
# #                          camera_azth=40.47,
# #                          camera_elev=32.5,
# #                          camera_fov=35.0,
# #                          camera_loc=(40.85, 46.32, 28.85),
# #                          camera_zoom=0.005124,
# #                          cb_x_offset=5,
# #                          width=620)
