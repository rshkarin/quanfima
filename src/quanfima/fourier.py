import math

from skimage import measure, filters
from skimage.util.shape import view_as_blocks
from scipy import ndimage as ndi
import numpy as np


def estimate_2d_orientation_in_blocks(data, grid_shape=(2, 2), sigma=2.0, zoom=1.0, order=3):
    """Computes orientation at every block of the subdivided image.

    Subdivides the image into the grid of blocks `grid_shape`, it computes
    2D FFT for every block, then the real components are segmented by
    the Otsu thresholding and the PCA-based approach calculates the orientation
    of structures within each block.

    Parameters
    ----------
    data : ndarray
        Indicates the grayscale 2D image.

    grid_shape : tuple
        Indicates the number of blocks to subdivide the image.

    sigma : float
        Indicates the sigma value of the Gaussian filter to smooth the real part
        of 2D Fourier spectrum.

    zoom : float
        Indicates the upscaling factor of each block before applying 2D FFT.

    order : str
        Indicates the order of interpolation used in the upscaling procedure.

    Returns
    -------
    (orient_blocks, block_shape) : tuple of arrays
        The 2D array of the orientation angle within each block,
        the shape of a block.
    """
    block_shape = tuple(
        [int(math.floor(d / float(gs))) for d, gs in zip(data.shape, grid_shape)]
    )
    data_blocks = view_as_blocks(data, block_shape=block_shape)
    orient_blocks = np.zeros(grid_shape, dtype=np.float32)

    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            dblock = data_blocks[i, j]

            if zoom != 1.0:
                dblock = ndi.interpolation.zoom(dblock, zoom, order=order)

            dblock_freq = np.abs(np.fft.fftshift(np.fft.fft2(dblock)).real)
            dblock_freq = ndi.gaussian_filter(dblock_freq, sigma=sigma)
            dblock_mask = (dblock_freq > filters.threshold_otsu(dblock_freq)).astype(
                np.uint8
            )

            lbls = ndi.label(
                dblock_mask, structure=ndi.generate_binary_structure(2, 2)
            )[0]
            rgns = measure.regionprops(lbls)

            dists = [(r.label, r.area) for r in rgns]
            dists = sorted(dists, key=lambda x: x[1])

            orient_blocks[i, j] = [
                r.orientation for r in rgns if r.label == dists[0][0]
            ][0]

    return (orient_blocks, block_shape)
