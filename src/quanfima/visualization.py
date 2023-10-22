from __future__ import print_function
import os
import math
import numpy as np
import matplotlib as mpl
from skimage import io, filters
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mpl_toolkits.axisartist.floating_axes as floating_axes
from matplotlib import cm
from scipy import ndimage as ndi, interpolate
from skimage import morphology
from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D
from mpl_toolkits.axisartist.grid_finder import DictFormatter, FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axisartist.floating_axes import GridHelperCurveLinear
from utils import geo2rgb
from quanfima import visvis_available
if visvis_available:
    import visvis as vv


def plot_orientation_map(orient_map, fiber_skel, radius_structure_elem=1,
                         figsize=(12, 12), cmap='hsv', dpi=200, min_label='0',
                         max_label='180', name=None, output_dir=None):
    """Plots the orientation map with the color wheel.

    Plots the orientation map from the provided angles `orient_map` and skeleton
    `fiber_skel` of size `figsize` using the colormap `cmap` and writes as a png
    file with DPI of `dpi` to the folder specified by `output_dir`.

    Parameters
    ----------
    orient_map : ndarray
        2D array of orientation at every point of the skeleton.

    fiber_skel : ndarray
        The binary skeleton extracted from the binary data.

    radius_structure_elem : integer
        Indicates the size of the structure element of the dilation process to
        thicken the skeleton.

    figsize : tuple of integers
        Indicates the size of the output figure.

    cmap : str
        Indicates the name of a colormap used to map angles to colors.

    dpi : integer
        Indicates the DPI of the output image.

    min_label : str
        Indicates the label of minimum degree.

    max_label : str
        Indicates the label of minimum degree.

    name : str
        Indicates the name of the output png file.

    output_dir : str
        Indicates the path to the output folder where the image will be stored.
    """
    disk = morphology.disk(radius_structure_elem)
    orient_map = ndi.grey_dilation(orient_map, structure=disk).astype(np.float32)
    fiber_skel = ndi.binary_dilation(fiber_skel, structure=disk).astype(np.float32)

    fig = plt.figure(figsize=figsize)
    masked_orient_map = np.ma.masked_where(fiber_skel == 0, orient_map)

    quant_steps = 2056
    cmap_obj = cm.get_cmap(cmap, quant_steps)
    cmap_obj.set_bad(color='black')

    omap_ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    omap_ax.set_axis_off()
    omap_ax.imshow(masked_orient_map, cmap=cmap_obj, interpolation=None)

    display_axes = fig.add_axes([0.780, -0.076, 0.2, 0.2], projection='polar')
    display_axes._direction = np.pi

    norm = mpl.colors.Normalize(0.0, np.pi)
    cb = mpl.colorbar.ColorbarBase(display_axes, cmap=cmap_obj, norm=norm, orientation='horizontal')

    display_axes.text(0.09, 0.56, min_label, color='white', fontsize=20, weight='bold', horizontalalignment='center',
                 verticalalignment='center', transform=display_axes.transAxes)

    display_axes.text(0.85, 0.56, max_label, color='white', fontsize=20, weight='bold', horizontalalignment='center',
                 verticalalignment='center', transform=display_axes.transAxes)

    cb.outline.set_visible(False)
    display_axes.set_axis_off()

    if (output_dir is not None) and (name is not None):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fig.savefig(os.path.join(output_dir, '{}_orientation_map.png'.format(name)),
                    transparent=False,
                    bbox_inches='tight',
                    pad_inches=0.1,
                    dpi=dpi)
        
    plt.show()


def plot_diameter_map(thickness_map, fiber_skel, radius_structure_elem=1,
                      figsize=(15, 15), cmap='hsv', tmin=None, tmax=None, dpi=200,
                      labelsize=20, label='Diameter, [pixels]', name=None,
                      output_dir=None):
    """Plots the diameter map with the colorbar.

    Plots the diameter map from the provided diameters `thickness_map` and
    skeleton `fiber_skel` of size `figsize` using the colormap `cmap` and the
    limits of the colorbar specified by `tmin` and `tmax`, and writes as a png
    file with DPI of `dpi` to the folder specified by `output_dir`.

    Parameters
    ----------
    thickness_map : ndarray
        2D array of diameter at every point of the skeleton.

    fiber_skel : ndarray
        The binary skeleton extracted from the binary data.

    radius_structure_elem : integer
        Indicates the size of the structure element of the dilation process to
        thicken the skeleton.

    figsize : tuple of integers
        Indicates the size of the output figure.

    cmap : str
        Indicates the name of a colormap used to map angles to colors.

    tmin : float
        Indicates the minimum value of the colorbar.

    tmax : float
        Indicates the maximum value of the colorbar.

    dpi : integer
        Indicates the DPI of the output image.

    labelsize : integer
        Indicates the fontsize of the label of the colorbar.

    label : str
        Indicates the label of the colorbar.

    name : str
        Indicates the name of the output png file.

    output_dir : str
        Indicates the path to the output folder where the image will be stored.
    """
    disk = morphology.disk(radius_structure_elem)
    thickness_map = ndi.grey_dilation(thickness_map, structure=disk).astype(np.float32)
    fiber_skel = ndi.binary_dilation(fiber_skel, structure=disk).astype(np.float32)

    masked_thickness_map = np.ma.masked_where(fiber_skel == 0, thickness_map)

    cmap_obj = cm.get_cmap(cmap)
    cmap_obj.set_bad(color='black')

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    ax.set_axis_off()
    
    im = ax.imshow(masked_thickness_map, cmap=cmap_obj, vmin=tmin, vmax=tmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size="2.5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=labelsize)
    cbar.set_label(label, fontsize=labelsize)

    if (output_dir is not None) and (name is not None):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.savefig(os.path.join(output_dir, '{}_diameter_map.png'.format(name)),
                    transparent=False,
                    bbox_inches='tight',
                    pad_inches=0.1,
                    dpi=dpi)
    plt.show()


def gather_polar_errors(datasets_path, estimation_path,
                        azth_rng=np.arange(-90, 91, step=3),
                        lat_rng=np.arange(0, 91, step=3)):
    """Computes the absolute angular error in ranges between estimated datasets.

    Calculates the absolute angular error in ranges `azth_rng` and `lat_rng`
    between estimated orientation datasets localed at paths `datasets_path` and
    `estimation_path`.

    Parameters
    ----------
    datasets_path : str
        Indicates the path to the reference / estimated dataset.

    estimation_path : str
        Indicates the path to the estimated / reference dataset.

    azth_rng : array
        Indicates the ranges of azimuth angles where the error is accumulated.

    lat_rng : array
        Indicates the ranges of latitude or elevation angles where the error
        is accumulated.

    Returns
    -------
    array : 2D array
        The 2D array of accumulated errors within the combinations of the specified
        angular ranges.
    """
    reference_dataset = np.load(datasets_path).item()
    estimated_dataset = np.load(estimation_path).item()
    idxs = estimated_dataset['indices']

    print(reference_dataset.keys())
    print(estimated_dataset.keys())

    ref_azth, ref_lat = reference_dataset['azth'][idxs], reference_dataset['lat'][idxs]
    res_azth, res_lat = estimated_dataset['azth'][idxs], estimated_dataset['lat'][idxs]

    abs_err_azth, abs_err_lat = np.abs(ref_azth - res_azth), np.abs(ref_lat - res_lat)

    out = _angle_err2mean_abs_err(abs_err_azth, abs_err_lat,
                                  ref_azth, ref_lat,
                                  azth_rng=azth_rng, lat_rng=lat_rng)

    return out


def _angle_err2mean_abs_err(azth_err, lat_err, azth_ref, lat_ref,
                            azth_rng=np.arange(-90, 91, step=5),
                            lat_rng=np.arange(0, 91, step=5)):
    """Computes the absolute angular error in ranges between estimated datasets.

    Parameters
    ----------
    azth_err : 3D array
        Indicates the error of azimuth components.

    lat_err : 3D array
        Indicates the error of latitude / elevation components.

    azth_ref : 3D array
        Indicates the reference dataset of estimated azimuth component.

    lat_ref : 3D array
        Indicates the reference dataset of estimated latitude or elevation component.

    azth_rng : array
        Indicates the ranges of azimuth angles where the error is accumulated.

    lat_rng : array
        Indicates the ranges of latitude and elevation angles where the error
        is accumulated.

    Returns
    -------
    array : 2D array
        The 2D array of accumulated errors within the combinations of the specified
        angular ranges.
    """
    out = np.zeros((len(azth_rng) - 1, len(lat_rng) - 1), dtype=np.float32)

    for i in xrange(len(azth_rng) - 1):
        for j in xrange(len(lat_rng) - 1):
            rng_azth = (np.deg2rad(azth_rng[i]), np.deg2rad(azth_rng[i+1]))
            rng_lat = (np.deg2rad(lat_rng[j]), np.deg2rad(lat_rng[j+1]))

            idxs_azth = np.where((azth_ref >= rng_azth[0]) & (azth_ref < rng_azth[1]))
            idxs_lat = np.where((lat_ref >= rng_lat[0]) & (lat_ref < rng_lat[1]))

            idxs = np.intersect1d(idxs_azth, idxs_lat)

            if len(idxs):
                mu_azth, mu_lat = np.rad2deg(np.mean(azth_err[idxs])), \
                                  np.rad2deg(np.mean(lat_err[idxs]))

                out[i, j] = mu_azth + mu_lat
            else:
                out[i, j] = 0.0

    return out


def plot_polar_heatmap(data, name, interp_factor=5., color_limits=False,
                       hide_colorbar=False, vmin=None, vmax=None, log_scale=True,
                       dpi=200, output_dir=None):
    """Plots the polar heatmap describing azimuth and latitude / elevation components.

    Plots the polar heatmap where each cell of the heatmap corresponds to
    the specific element of the array provided by `gather_polar_errors`
    function.

    Parameters
    ----------
    data : 2D array
        Indicates the array containing the sum of angular errors within the
        specified angular ranges. It is usually provided by `gather_polar_errors`
        function.

    name : str
        Indicates the name of the output png file.

    interp_factor : float
        Indicates the interpolation factor of the heatmap.

    color_limits : boolean
        Specifies if the determined intensity limits should be returned.

    hide_colorbar : boolean
        Specifies if the colorbar should be hidden.

    vmin : float
        Indicates the minimum value of the colorbar.

    vmax : float
        Indicates the maximum value of the colorbar.

    log_scale : float
        Specifies if the heatmap sould be in the logarithmic scale.

    dpi : integer
        Indicates the DPI of the output image.

    output_dir : str
        Indicates the path to the output folder where the image will be stored.
    """
    th0, th1 = 0., 180.
    r0, r1 = 0, 90
    thlabel, rlabel = 'Azimuth', 'Elevation'

    tr_scale = Affine2D().scale(np.pi/180., 1.)
    tr = tr_scale + PolarAxes.PolarTransform()

    lat_ticks = [(.0*90., '0$^{\circ}$'),
                 (.33*90., '30$^{\circ}$'),
                 (.66*90., '60$^{\circ}$'),
                 (1.*90., '90$^{\circ}$')]
    r_grid_locator = FixedLocator([v for v, s in lat_ticks])
    r_grid_formatter = DictFormatter(dict(lat_ticks))

    angle_ticks = [(0*180., '90$^{\circ}$'),
                   (.25*180., '45$^{\circ}$'),
                   (.5*180., '0$^{\circ}$'),
                   (.75*180., '-45$^{\circ}$'),
                   (1.*180., '-90$^{\circ}$')]
    theta_grid_locator = FixedLocator([v for v, s in angle_ticks])
    theta_tick_formatter = DictFormatter(dict(angle_ticks))

    grid_helper = GridHelperCurveLinear(tr,
                                        extremes=(th0, th1, r0, r1),
                                        grid_locator1=theta_grid_locator,
                                        grid_locator2=r_grid_locator,
                                        tick_formatter1=theta_tick_formatter,
                                        tick_formatter2=r_grid_formatter)

    fig = plt.figure()
    ax = floating_axes.FloatingSubplot(fig, 111, grid_helper=grid_helper)
    fig.add_subplot(ax)
    ax.set_facecolor('white')

    ax.axis["bottom"].set_visible(False)
    ax.axis["top"].toggle(ticklabels=True, label=True)
    ax.axis["top"].set_axis_direction("bottom")
    ax.axis["top"].major_ticklabels.set_axis_direction("top")
    ax.axis["top"].label.set_axis_direction("top")

    ax.axis["left"].set_axis_direction("bottom")
    ax.axis["right"].set_axis_direction("top")

    ax.axis["top"].label.set_text(thlabel)
    ax.axis["left"].label.set_text(rlabel)

    aux_ax = ax.get_aux_axes(tr)
    aux_ax.patch = ax.patch
    ax.patch.zorder = 0.9

    rad = np.linspace(0, 90, data.shape[1])
    azm = np.linspace(0, 180, data.shape[0])

    f = interpolate.interp2d(rad, azm, data, kind='linear', bounds_error=True, fill_value=0)

    new_rad = np.linspace(0, 90, 180*interp_factor)
    new_azm = np.linspace(0, 180, 360*interp_factor)
    new_data_angle_dist = f(new_rad, new_azm)
    new_r, new_th = np.meshgrid(new_rad, new_azm)
    new_data_angle_dist += 1.

    if log_scale:
        data_mesh = aux_ax.pcolormesh(new_th, new_r, new_data_angle_dist, cmap='jet',
                                      norm=colors.LogNorm(vmin=1. if vmin is None else vmin,
                                                          vmax=new_data_angle_dist.max() if vmax is None else vmax))
    else:
        data_mesh = aux_ax.pcolormesh(new_th, new_r, new_data_angle_dist, cmap='jet', vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(data_mesh, orientation='vertical', shrink=.88, pad=.1, aspect=15)
    cbar.ax.set_ylabel('Absolute error, [deg.]')

    if hide_colorbar:
        cbar.remove()

    ax.grid(False)

    plt.show()

    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fig.savefig(os.path.join(output_dir, '{}_chart.png'.format(name)),
                    transparent=False,
                    bbox_inches='tight',
                    pad_inches=0.1,
                    dpi=dpi)

    if color_limits:
        return 1., new_data_angle_dist.max()


def plot_histogram_fig(data, num_bins, xticks, color, splot_index=111, output_dir=None,
                       xlim=(None, None), ylim=(None, None), name=None, in_percent=False,
                       bar_width=0.8, ticks_pad=7, xticks_fontsize=22,
                       yticks_fontsize=22, xlabel=None, ylabel=None,
                       labels_fontsize=20, grid_alpha=0.3, title_fontsize=22,
                       exp_fontsize=15, type=None, figsize=(12, 8), dpi=200):
    """Plots the histogram from a given data.

    Parameters
    ----------
    data : 1D array
        Indicates the array containing the values.

    num_bins : integer
        Indicates the number of histogram bins.

    xticks : array
        Indicates the array of ticks of the X-axis.

    color : str
        Indicates the color of the histogram.

    Returns
    -------
    tuple : tuple of Figure and Axis objects.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(splot_index)

    weights = np.ones_like(data)/float(len(data))

    n, bins, patches = ax.hist(data, num_bins, color=color,
                               rwidth=bar_width, weights=weights)

    ax.tick_params(axis='x',
                   labelsize=xticks_fontsize,
                   colors='#000000',
                   which='both',
                   direction='out',
                   length=6,
                   width=2,
                   pad=ticks_pad)
    ax.tick_params(axis='x',
                   labelsize=xticks_fontsize,
                   colors='#000000',
                   which='minor',
                   direction='out',
                   length=4,
                   width=2,
                   pad=ticks_pad)

    ax.tick_params(axis='y',
                   labelsize=yticks_fontsize,
                   colors='#000000',
                   which='major',
                   direction='out',
                   length=6,
                   width=2,
                   pad=ticks_pad)
    ax.tick_params(axis='y',
                   labelsize=xticks_fontsize,
                   colors='#000000',
                   which='minor',
                   direction='out',
                   length=4,
                   width=2,
                   pad=ticks_pad)

    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))

    plt.xticks(xticks, xticks)

    ax.xaxis.offsetText.set_fontsize(exp_fontsize)
    ax.yaxis.offsetText.set_fontsize(exp_fontsize)

    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2),
                        fontsize=yticks_fontsize)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, min_n_ticks=6))

    ax.xaxis.grid(False)
    ax.yaxis.grid(False)

    ax.set_ylabel(ylabel, labelpad=2, fontsize=labels_fontsize, color='black')
    ax.set_xlabel(xlabel, labelpad=2, fontsize=labels_fontsize, color='black')

    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

    if in_percent:
        vals = ax.get_yticks()
        ax.set_yticklabels(['{:3.0f}'.format(x*100) for x in vals])

    ax.set_axisbelow(True)

    if (output_dir is not None) and (name is not None):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, '{}_chart.png'.format(name)),
                    transparent=False, bbox_inches='tight', pad_inches=0.1, dpi=dpi)

    return fig, ax


def create_pie_chart(data, rngs, colors=['#244268', '#426084', '#67809F', '#95A9C1', '#C6D2E0'],
                     unit_scale=1.0, measure_quantity='m^3', figsize=(33, 15),
                     legend_loc=(0.383, -0.25), zebra_color=(False, 3),
                     legend_fontsize=50, chart_fontsize=60, dpi=72, name=None,
                     output_dir=None):
    """Plots the piechart of from a given data.

    Parameters
    ----------
    data : 1D array
        Indicates the array containing the values.

    rngs : tuple of tuples
        Indicates the ranges of the piechart.

    colors : array
        Indicates the color for the region of the piechart corresponding to the
        specific range.

    unit_scale : float
        Indicates the scale factor of the data values.

    measure_quantity : str
        Indicates the name of measure of the values.

    figsize : tuple of integers
        Indicates the size of the output figure.

    legend_loc : tuple
        Indicates the position of the legend of the figure.

    zebra_color : tuple
        Allows to change the text color of the region to white from the first to
        the speficied index of the region (True, reg_index).

    legend_fontsize : integer
        Indicates the fontsize of the legend.

    chart_fontsize : integer
        Indicates the fontsize of the figure.

    dpi : integer
        Indicates the DPI of the output image.

    name : str
        Indicates the name of the output png file.

    output_dir : str
        Indicates the path to the output folder where the image will be stored.
    """
    def number(val):
        if val < 1000:
            return '%d' % val

        sv = str(val)
        return '$\mathregular{10^{%d}}$' % (len(sv)-2) if val % 10 == 0 else '%0.0e' % val

    def get_title(v1, v2, measure_quantity):
        ftm = '%s $\minus$ %s %s'
        return ftm % (number(v1), number(v2), measure_quantity)

    data_ranges = []
    df = data * unit_scale
    for rng in rngs:
        rng_min, rng_max = rng[0], rng[1]
        data_rng = df[(df > rng_min) & (df < rng_max)]
        data_ranges.append(data_rng)

    num_elem = [len(p) for p in data_ranges]
    se = sum(num_elem)

    print('Num of particles: {}'%(se))

    proc_particles = [n/float(se) * 100.0 for n in num_elem]

    for size, rng in zip(num_elem, rngs):
        print('{}-{}: {}'%(rng[0], rng[1], size))

    titles = [get_title(minv, maxv, measure_quantity) for minv, maxv in rngs]

    textprops = {'fontsize': chart_fontsize,
                 'weight': 'normal',
                 'family': 'sans-serif'}
    pie_width = 0.5
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('equal')
    patches, texts, autotexts = ax.pie(proc_particles,
                                       textprops=textprops,
                                       colors=colors,
                                       autopct='%1.1f%%',
                                       radius=1,
                                       pctdistance=1-pie_width/2)

    if (zebra_color is not None) and (zebra_color[0]):
        for tt in autotexts[:zebra_color[1]]:
            tt.set_color('white')

    plt.setp(patches,
             width=pie_width,
             edgecolor='white')

    plt.legend(patches, titles, loc=legend_loc, fontsize=legend_fontsize)

    _d, _offset, _di = [1, -1], [0.45, 0.45], 0

    for t, p in zip(autotexts, proc_particles):
        if p < 2.0:
            pos = list(t.get_position())
            pos[0] = pos[0] + _d[_di] * _offset[_di]

            t.set_position(pos)
            _di += 1

    if (output_dir is not None) and (name is not None):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, '{}_chart.png'.format(name)),
                    bbox_inches='tight', transparent=True, pad_inches=0.1, dpi=dpi)


def _bbox_3D(img):
    """Crops the non-zero part of a volume
    """
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax


def plot_3d_orientation_map(name, lat_data, azth_data, radius_structure_elem=1,
                            output_dir=None, width=512, height=512,
                            camera_azth=44.5, camera_elev=35.8, camera_roll=0.0,
                            camera_fov=35.0, camera_zoom=0.0035,
                            camera_loc=(67.0, 81.6, 45.2), xlabel='', ylabel='',
                            zlabel='', axis_color='w', background_color='k'):
    """Renders orientation data in 3D with RGB angular color-coding.

    Parameters
    ----------
    name : str
        Indicates the name of the output png file.

    lat_data : 3D array
        Indicates the 3D array containing latitude / elevation angle at every point of
        the skeleton in radians.

    azth_data : 3D array
        Indicates the 3D array containing azimuth angle at every point of the skeleton
        in radians.

    radius_structure_elem : integer
        Indicates the size of the structure element of the dilation process to
        thicken the skeleton.

    output_dir : str
        Indicates the path to the output folder where the image will be stored.

    width : int
        Indicates the width of the visualization window.

    height : int
        Indicates the width of the visualization window.

    camera_azth : float
        Indicates the azimuth angle of the camera.

    camera_elev : float
        Indicates the latitude / elevation angle of the camera.

    camera_roll : float
        Indicates the roll angle of the camera.

    camera_fov : float
        Indicates the field of view of the camera.

    camera_zoom : float
        Indicates the zoom level of the camera.

    camera_loc : tuple
        Indicates the camera location.

    xlabel : str
        Indicates the label along the x-axis.

    ylabel : str
        Indicates the label along the y-axis.

    zlabel : str
        Indicates the label along the z-axis.

    axis_color : str
        Indicates the color of axes.

    background_color : str
        Indicates the background color of the figure.
    """
    if not visvis_available:
        print('The visvis package is not found. The visualization cannot be done.')
        return

    rmin, rmax, cmin, cmax, zmin, zmax = _bbox_3D(azth_data)

    azth, lat = azth_data[rmin:rmax, cmin:cmax, zmin:zmax], \
                np.abs(lat_data[rmin:rmax, cmin:cmax, zmin:zmax])

    skel = azth.copy().astype(np.float32)
    skel[skel.nonzero()] = 1.

    azth = ndi.grey_dilation(azth, structure=morphology.ball(radius_structure_elem))
    lat = ndi.grey_dilation(lat, structure=morphology.ball(radius_structure_elem))
    skel = ndi.binary_dilation(skel, structure=morphology.ball(radius_structure_elem))

    Z, Y, X = skel.nonzero()
    vol_orient = np.zeros(skel.shape + (3,), dtype=np.float32)

    print(vol_orient.size, vol_orient[skel.nonzero()].size)

    for z, y, x in zip(Z, Y, X):
        vol_orient[z, y, x] = geo2rgb(lat[z, y, x], azth[z, y, x])

    app = vv.use()

    fig = vv.figure()
    fig._currentAxes = None
    fig.relativeFontSize = 2.
    fig.position.w = width
    fig.position.h = height

    t = vv.volshow(vol_orient[:, :, :], renderStyle='iso')
    t.isoThreshold = 0.5

    a = vv.gca()
    a.camera.azimuth = camera_azth
    a.camera.elevation = camera_elev
    a.camera.roll = camera_roll
    a.camera.fov = camera_fov
    a.camera.zoom = camera_zoom
    a.camera.loc = camera_loc

    a.bgcolor = background_color
    a.axis.axisColor = axis_color
    a.axis.xLabel = xlabel
    a.axis.yLabel = ylabel
    a.axis.zLabel = zlabel

    # def mouseUp(event):
    #     print 'mouseUp!!'
    #     a = vv.gca()
    #     print a.camera.GetViewParams()
    #
    # a.eventMouseUp.Bind(mouseUp)
    # fig.eventMouseUp.Bind(mouseUp)
    #
    # a.Draw()
    # fig.DrawNow()

    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        vv.screenshot(os.path.join(output_dir, '{}_3d_orientation.png'.format(name)),
                      sf=1, bg=background_color)
    app.Run()


def plot_3d_diameter_map(name, data, unit_scale=1.0, measure_quantity='vox',
                         radius_structure_elem=1, output_dir=None, width=512,
                         height=512, camera_azth=44.5, camera_elev=35.8,
                         camera_roll=0.0, camera_fov=35.0, camera_zoom=0.0035,
                         camera_loc=(67.0, 81.6, 45.2), xlabel='', ylabel='',
                         zlabel='', axis_color='w', background_color='k',
                         cb_x_offset=10):
    """Renders orientation data in 3D with RGB angular color-coding.

    Parameters
    ----------
    name : str
        Indicates the name of the output png file.

    data : 3D array
        Indicates the 3D array containing diameter at every point of the skeleton.

    unit_scale : float
        Indicates the scale factor of the data values.

    measure_quantity : str
        Indicates the name of measure of the values.

    radius_structure_elem : integer
        Indicates the size of the structure element of the dilation process to
        thicken the skeleton.

    output_dir : str
        Indicates the path to the output folder where the image will be stored.

    camera_azth : float
        Indicates the azimuth angle of the camera.

    width : int
        Indicates the width of the visualization window.

    height : int
        Indicates the width of the visualization window.

    camera_elev : float
        Indicates the latitude / elevation angle of the camera.

    camera_roll : float
        Indicates the roll angle of the camera.

    camera_fov : float
        Indicates the field of view of the camera.

    camera_zoom : float
        Indicates the zoom level of the camera.

    camera_loc : tuple
        Indicates the camera location.

    xlabel : str
        Indicates the label along the x-axis.

    ylabel : str
        Indicates the label along the y-axis.

    zlabel : str
        Indicates the label along the z-axis.

    axis_color : str
        Indicates the color of axes.

    background_color : str
        Indicates the background color of the figure.

    cb_x_offset : int
        Indicates the offset of the colorbar from the right window side.
    """
    if not visvis_available:
        print('The visvis package is not found. The visualization cannot be done.')
        return

    rmin, rmax, cmin, cmax, zmin, zmax = _bbox_3D(data)
    dmtr = data[rmin:rmax, cmin:cmax, zmin:zmax] * unit_scale
    skel = np.zeros_like(dmtr, dtype=np.uint8)
    skel[dmtr.nonzero()] = 1

    dmtr = ndi.grey_dilation(dmtr, structure=morphology.ball(radius_structure_elem))
    skel = ndi.binary_dilation(skel, structure=morphology.ball(radius_structure_elem)).astype(np.float32)
    skel[skel.nonzero()] = 1.

    dmtr = dmtr * skel

    app = vv.use()

    fig = vv.figure()
    fig._currentAxes = None
    fig.relativeFontSize = 2.
    fig.position.w = width
    fig.position.h = height

    t = vv.volshow(dmtr[:, :, :], renderStyle='iso')
    t.isoThreshold = 0.5
    t.colormap = vv.CM_JET

    a = vv.gca()
    a.camera.azimuth = camera_azth
    a.camera.elevation = camera_elev
    a.camera.roll = camera_roll
    a.camera.fov = camera_fov
    a.camera.zoom = camera_zoom
    a.camera.loc = camera_loc

    a.bgcolor = background_color
    a.axis.axisColor = axis_color
    a.axis.xLabel = xlabel
    a.axis.yLabel = ylabel
    a.axis.zLabel = zlabel

    cb = vv.colorbar()
    cb.SetLabel('Diameter, [{}]'.format(measure_quantity))
    cb._label.position.x += cb_x_offset

    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        vv.screenshot(os.path.join(output_dir, '{}_3d_diameter.png'.format(name)),
                      sf=1, bg='w')
    app.Run()


def plot_color_wheel(name, output_dir=None, dpi=500, xlabel='Elevation',
                     ylabel='Azimuth', fontsize=10, num_xticks=4, yticks=(-90, 90)):
    """Plots the color wheel for visualizations of 3D orintation.

    Parameters
    ----------
    name : str
        Indicates the name of the output png file.

    output_dir : str
        Indicates the path to the output folder where the image will be stored.

    dpi : integer
        Indicates the DPI of the output image.

    xlabel : str
        Indicates the text along the x-axis.

    ylabel : str
        Indicates the text along the y-axis.

    fontsize : int
        Indicates the font size of labels along axes.

    num_xticks : int
        Indicates the number of ticks along axes.

    yticks : tuple
        Indicates the range of minimum and maximum values along the y-axis.
    """

    azth, lat = np.linspace(0., 1., num=180), np.linspace(0., 1., num=90)
    rgb_arr = np.zeros((len(azth), len(lat), 3))
    for i in xrange(len(azth)):
        for j in xrange(len(lat)):
            rgb_arr[i, j, :] = colors.hsv_to_rgb([azth[i], lat[j], 1.0])

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.set_facecolor('red')

    ax.set_xlim([0, 90])
    ax.set_ylim([0, 181])

    ax.set_yticks(np.linspace(0, 180, num=7).astype(np.int32))
    ax.set_yticklabels(np.linspace(yticks[0], yticks[1], num=7).astype(np.int32))

    ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=0, color='w')
    ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=0, color='w')

    xmajorlocator = ticker.LinearLocator(num_xticks)
    ax.xaxis.set_major_locator(xmajorlocator)
    ax.tick_params(direction='out', length=2, width=0, labelsize=fontsize, pad=0, colors='w')

    ax.imshow(rgb_arr)

    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fig.savefig(os.path.join(output_dir, '{}_color_bar.png'.format(name)),
                    transparent=True,
                    bbox_inches='tight',
                    pad_inches=0.1,
                    dpi=dpi)
    plt.show()

def plot_fourier_orientation(data, orient_blocks, block_shape, figsize=(12,12),
                             cmap='gray', line_length=20, line_width=2.5,
                             line_color='red', line_style='-', name=None,
                             output_dir=None, dpi=200):
    """Plots the orientation map in a block-wise manner.

    Plots the orientation vector over the image `data` at the center of each
    block of the subdivided image. The orientation at every block 'orient_blocks'
    and its size `block_shape` are specified by orientation estimation method.
    The result can be and stored as a png file with DPI of `dpi` to the folder
    specified by `output_dir`.

    Parameters
    ----------
    data : ndarray
        An image on top of which, the orientation will be plotted..

    orient_blocks : ndarray
        2D array of orientation at every block of the subdivided image.

    block_shape : tuple
        Indicates the block size within which the orientation is calculated.

    figsize : tuple of integers
        Indicates the size of the output figure.

    cmap : str
        Indicates the name of a colormap used for image.

    line_length : integer
        Indicates the length of the orientation vector at each block.

    line_width : float
        Indicates the line width of the orientation vector at each block.

    line_color : str
        Indicates the line color of the orientation vector at each block.

    line_style : str
        Indicates the line style of the orientation vector at each block.

    name : str
        Indicates the name of the output png file.

    output_dir : str
        Indicates the path to the output folder where the image will be stored.

    dpi : integer
        Indicates the DPI of the output image.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_axis_off()

    ax.imshow(data, cmap=cmap)

    for i in xrange(orient_blocks.shape[0]):
        for j in xrange(orient_blocks.shape[1]):
            y0, x0 = block_shape[0] * j + block_shape[0]/2, \
                     block_shape[1] * i + block_shape[1]/2

            orientation = orient_blocks[i ,j]

            x2 = x0 - math.sin(orientation) * line_length
            y2 = y0 - math.cos(orientation) * line_length

            x3 = x0 + math.sin(orientation) * line_length
            y3 = y0 + math.cos(orientation) * line_length

            ax.plot((x2, x3), (y2, y3), linestyle=line_style,
                    linewidth=line_width, color=line_color)

    if (output_dir is not None) and (name is not None):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, '{}_fourier_orientation_map.png'.format(name)),
                    bbox_inches='tight', transparent=True, pad_inches=0.05, dpi=dpi)

    plt.show()
