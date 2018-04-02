==========
Quickstart
==========

Analysis of fibrous 2D data
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Open a grayscale image, perform segmentation, estimate porosity, analyze fiber
orientation and diameters, and plot the results.

.. code-block:: python

  import numpy as np
  from skimage import io, filters
  from quanfima import morphology as mrph
  from quanfima import visualization as vis
  from quanfima import utils

  img = io.imread('../../data/polymer_slice.tif')

  th_val = filters.threshold_otsu(img)
  img_seg = (img > th_val).astype(np.uint8)

  # estiamte porosity
  pr = mrph.calc_porosity(img_seg)
  for k,v in pr.items():
    print 'Porosity ({}): {}'.format(k, v)

  # prepare data and analyze fibers
  data, skeleton, skeleton_thick = utils.prepare_data(img_seg)
  cskel, fskel, omap, dmap, ovals, dvals = \
                      mrph.estimate_fiber_properties(data, skeleton)

  # plot results
  vis.plot_orientation_map(omap, fskel, min_label=u'0°', max_label=u'180°',
                           figsize=(10,10),
                           name='2d_polymer',
                           output_dir='../../data/results')
  vis.plot_diameter_map(dmap, cskel, figsize=(10,10), cmap='gist_rainbow',
                        name='2d_polymer',
                        output_dir='../../data/results')

.. code-block:: python

  >> Porosity (Material 1): 0.845488888889

.. image:: _static\2d_polymer_data.png
    :width: 90 %
    :align: center
.. image:: _static\2d_polymer_orientation_map.png
    :width: 90 %
    :align: center
.. image:: _static\2d_polymer_diameter_map.png
    :width: 90 %
    :align: center

Analysis of 3D data of fibrous material
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Open a micro-CT dataset, perform slice-wise segmentation, estimate porosity,
analyze 3D fiber orientation and diameters, and visualize the results.

.. code:: python

    import numpy as np
    from skimage import filters
    from quanfima import morphology as mrph
    from quanfima import visualization as vis
    from quanfima import utils

    data = np.memmap('../../data/polymer3d_8bit_128x128x128.raw',
                     shape=(128,128,128), dtype=np.uint8, mode='r')

    data_seg = np.zeros_like(data, dtype=np.uint8)
    for i in xrange(data_seg.shape[0]):
      th_val = filters.threshold_otsu(data[i])
      data_seg[i] = (data[i] > th_val).astype(np.uint8)

    # estimate porosity
    pr = mrph.calc_porosity(data_seg)
    for k,v in pr.items():
      print 'Porosity ({}): {}'.format(k, v)

    # prepare data and analyze fibers
    pdata, pskel, pskel_thick = utils.prepare_data(data_seg)
    oprops =  mrph.estimate_tensor_parallel('polymer_orientation_w32', pskel,
                                            pskel_thick, 32,
                                            '../../data/results')

    odata = np.load(oprops['output_path']).item()
    lat, azth, skel = odata['lat'], odata['azth'], odata['skeleton']

    dprops = mrph.estimate_diameter_single_run('polymer_diameter',
                                               '../../data/results',
                                               pdata, skel, lat, azth)
    dmtr = np.load(dprops['output_path']).item()['diameter']

    # plot results
    vis.plot_3d_orientation_map('polymer_w32', lat, azth,
                                output_dir='../../data/results',
                                camera_azth=40.47,
                                camera_elev=32.5,
                                camera_fov=35.0,
                                camera_loc=(40.85, 46.32, 28.85),
                                camera_zoom=0.005124)

    vis.plot_3d_diameter_map('polymer_w32', dmtr,
                             output_dir='../../data/results',
                             measure_quantity='vox',
                             camera_azth=40.47,
                             camera_elev=32.5,
                             camera_fov=35.0,
                             camera_loc=(40.85, 46.32, 28.85),
                             camera_zoom=0.005124,
                             cb_x_offset=5,
                             width=620)

.. code:: python

    >> Porosity (Material 1): 0.855631351471

.. image:: _static\polymer_w32_3d_orientation.png
    :width: 78 %
    :align: center
.. image:: _static\polymer_w32_3d_diameter.png
    :width: 80 %
    :align: center

Estimation of p-values
^^^^^^^^^^^^^^^^^^^^^^
Estimate p-values between several groups of samples with corresponding
measurements of some material's property.

.. code:: python

    import pandas as pd
    from quanfima import utils

    prop_vals = [174.93, 182.42, 194.61, 234.6, 229.73, 242.6, 38.78, 37.79,
                 32.06, 14.81, 15.23, 13.84]
    mat_groups = ['PCL_cl', 'PCL_cl', 'PCL_cl', 'PCL_wa', 'PCL_wa', 'PCL_wa',
                  'PCL_SiHA_cl', 'PCL_SiHA_cl', 'PCL_SiHA_cl', 'PCL_SiHA_wa',
                  'PCL_SiHA_wa', 'PCL_SiHA_wa']
    df_elongation = pd.DataFrame({'elongation': prop_vals, 'type': mat_groups})

    _, _ = utils.calculate_tukey_posthoc(df_elongation, 'elongation',
                                         name='samples_elongation',
                                         write=True,
                                         output_dir='../../data/results')

.. code:: python

    >> Tukey post-hoc (elongation)
    >>     Multiple Comparison of Means - Tukey HSD,FWER=0.05
    >> =========================================================
    >>    group1      group2   meandiff  lower    upper   reject
    >> ---------------------------------------------------------
    >> PCL_SiHA_cl PCL_SiHA_wa -21.5833 -37.8384 -5.3282   True
    >> PCL_SiHA_cl    PCL_cl   147.7767 131.5216 164.0318  True
    >> PCL_SiHA_cl    PCL_wa   199.4333 183.1782 215.6884  True
    >> PCL_SiHA_wa    PCL_cl    169.36  153.1049 185.6151  True
    >> PCL_SiHA_wa    PCL_wa   221.0167 204.7616 237.2718  True
    >>    PCL_cl      PCL_wa   51.6567  35.4016  67.9118   True
    >> ---------------------------------------------------------
    >> ['PCL_SiHA_cl' 'PCL_SiHA_wa' 'PCL_cl' 'PCL_wa']
    >> PCL_SiHA_cl-PCL_SiHA_wa :  0.011919282004
    >> PCL_SiHA_wa-PCL_cl :  0.001
    >> PCL_SiHA_wa-PCL_wa :  0.001
    >> PCL_cl-PCL_wa :  0.001
    >> PCL_SiHA_cl-PCL_wa :  0.001
    >> PCL_SiHA_cl-PCL_cl :  0.001

Simulate and count particles in 3D data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Count and estimate properties of particles in a generated dataset comprised
of spheres of varying radius.

.. code:: python

    from quanfima import simulation
    from quanfima import morphology as mrph

    volume, diameter, _, _ = simulation.simulate_particles((512,512,512),
                                                            n_particles=1000)

    stats, labeled_volume = mrph.object_counter(volume)

.. code:: python

    >>     Label      Area    Perimeter  Sphericity
    >> 0      1.0   20479.0  2896.958728    1.249533
    >> 1      2.0    5575.0  1184.028571    1.284158
    >> 2      3.0   57777.0  5816.142853    1.242660
    >> 3      4.0   17077.0  2545.194226    1.260001
    >> 4      5.0    5575.0  1184.028571    1.284158
    >> 5      6.0   65267.0  6348.926691    1.234752
    >> ..     ...       ...          ...         ...
    >> 791  792.0    2109.0   605.185858    1.314154
    >> 792  793.0     257.0   134.225397    1.456369
    >> 793  794.0     257.0   134.225397    1.456369
    >> 794  795.0     123.0    78.627417    1.521179

    >> [795 rows x 4 columns]

Simulate fibers and estimate properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Simulate a 3D dataset containing some number of fibers, estimate their
properties and visualize.

.. code:: python

    import numpy as np
    from scipy import ndimage as ndi
    from skimage import morphology
    from quanfima import simulation
    from quanfima import morphology as mrph
    from quanfima import utils
    from quanfima import visualization as vis

    volume, lat_ref, azth_ref, diameter, _, _ = \
              simulation.simulate_fibers((128,128,128), n_fibers=30, max_fails=100,
                                         radius_lim=(2, 3), gap_lim=(3,5))
    volume = volume.astype(np.uint8)
    volume = ndi.binary_fill_holes(volume)
    volume = ndi.median_filter(volume, footprint=morphology.ball(2))
    lat_ref = ndi.median_filter(lat_ref, footprint=morphology.ball(2))
    azth_ref = ndi.median_filter(azth_ref, footprint=morphology.ball(2))

    # prepare data and analyze fibers
    pdata, pskel, pskel_thick = utils.prepare_data(volume)
    oprops =  mrph.estimate_tensor_parallel('dataset_orientation_w36',
                                            pskel, pskel_thick, 36,
                                            '../../data/results')

    odata = np.load(oprops['output_path']).item()
    lat, azth, skel = odata['lat'], odata['azth'], odata['skeleton']

    dprops = mrph.estimate_diameter_single_run('dataset_diameter',
                                               '../../data/results',
                                               pdata, skel, lat, azth)
    dmtr = np.load(dprops['output_path']).item()['diameter']

    # plot results
    vis.plot_3d_orientation_map('dataset_w36', lat, azth,
                                output_dir='../../data/results',
                                camera_azth=40.47,
                                camera_elev=32.5,
                                camera_fov=35.0,
                                camera_loc=(40.85, 46.32, 28.85),
                                camera_zoom=0.005124)

    vis.plot_3d_diameter_map('dataset_w36', dmtr,
                             output_dir='../../data/results',
                             measure_quantity='vox',
                             camera_azth=40.47,
                             camera_elev=32.5,
                             camera_fov=35.0,
                             camera_loc=(40.85, 46.32, 28.85),
                             camera_zoom=0.005124,
                             cb_x_offset=5,
                             width=620)

.. image:: _static\dataset_w36_3d_orientation.png
    :width: 78 %
    :align: center
.. image:: _static\dataset_w36_3d_diameter.png
    :width: 80 %
    :align: center
