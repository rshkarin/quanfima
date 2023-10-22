from __future__ import print_function
import os
import operator
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology
from matplotlib import colors
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.libqsturng import psturng


def prepare_data(data, dilate_iterations=1, sigma=0.5):
    """Returns the given binary data, its skeleton and the thickened skeleton.

    The skeleton of a given 2D or 3D array is computed, then it is thickened
    using morphological dilation with `dilate_iterations` and smoothed with
    help of Gaussian filter of specified `sigma`.

    Parameters
    ----------
    data : ndarray
        2D or 3D binary array which will be processed.

    dilate_iterations : integer
        Indicates the number of iterations for thickenning the skeleton.

    sigma : float
        Indicates the sigma of Gaussian filter used in smoothing of skeleton.

    Returns
    -------
    arrays : tuple of 2D or 3D arrays
        The original array, its skeleton and the thickened skeleton.
    """
    data_8bit = data.astype(np.uint8)
    data_8bit = ndi.binary_fill_holes(data_8bit).astype(np.uint8)

    if data.ndim == 3:
        skeleton = morphology.skeletonize_3d(data_8bit)
    elif data.ndim == 2:
        skeleton = morphology.skeletonize(data_8bit)
    else:
        raise ValueError('Incorrect number of data dimensions, it supports from 2 to 3 dimensions.')

    skeleton_thick = ndi.binary_dilation(skeleton, iterations=dilate_iterations).astype(np.float32)
    skeleton_thick = ndi.filters.gaussian_filter(skeleton_thick, sigma)

    return (data, skeleton, skeleton_thick)


def geo2rgb(lat, azth, azth_max=np.pi, lat_max=np.pi):
    """Translates geo-coordinates to color in RGB color space.

    Parameters
    ----------
    lat : float
        Indicates latitude or elevation component of a given geo-coordinates.

    azth : float
        Indicates azimuth component of a given geo-coordinates.

    azth_max : float
        Indicates the normalization value for the azimuth component.

    lat_max : float
        Indicates the normalization value for the latitude or elevation component.

    Returns
    -------
    array : tuple of values
        The tuple of RGB values [R, G, B].
    """
    return colors.hsv_to_rgb([azth/azth_max, lat/lat_max, 1.0])


def calculate_tukey_posthoc(df, column, type_column='type', verbose=True, write=False, name=None, output_dir=None):
    """Computes p-values using ANOVA with post-hoc Tukey HSD for a given DataFrame.

    Estimates p-values for a given DataFrame assuming that the sample
    type is named as `type_column`.

    Parameters
    ----------
    df : pandas DataFrame
        Contains the table of values ans corresponding types or classes.

    column : str
        Indicates the column of values.

    type_column : str
        Indicates the column of sample kind.

    verbose : boolean
        Specifies if the output should be printed into a terminal.

    write : boolean
        Specifies if the output should be written into a text file.

    name : str
        Indicates the name of the output file.

    output_dir : str
        Indicates the output dir where the file will be written.

    Returns
    -------
    dict : sample typles and p-values
        The dict of sample types and cooresponding p-values.
    """
    mc = MultiComparison(df[column], df[type_column])
    tt = mc.tukeyhsd()
    st_range = np.abs(tt.meandiffs) / tt.std_pairs

    fout = None
    if write and output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fout = open(os.path.join(output_dir, name + '.txt'), 'w')
        print(os.path.join(output_dir, name + '.txt'))

    if write:
        print('Tukey post-hoc ({0})'.format(column), end="", file=fout)
        print(tt, end="", file=fout)
        print(mc.groupsunique, end="", file=fout)

    if verbose:
        print('Tukey post-hoc ({0})'%(column))
        print(tt)
        print(mc.groupsunique)

    pvals = psturng(st_range, len(tt.groupsunique), tt.df_total)

    out = {}
    groups = mc.groupsunique
    g1idxs, g2idxs = mc.pairindices

    for g1i, g2i, p in zip(g1idxs, g2idxs, pvals):
        gname = '{}-{}'.format(groups[g1i], groups[g2i])
        out[gname] = p

    min_item = min(out.iteritems(), key=operator.itemgetter(1))

    for grp, p in out.items():
        if fout and write:
            print >> fout, '{}: {}'.format(grp, p)
        if verbose:
            print(grp, ': ', p)

    return out, min_item
