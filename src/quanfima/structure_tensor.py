import math

import numpy as np
from skimage import feature, measure
from scipy import ndimage as ndi
from scipy.spatial import distance


# 2D PCA-based approach
def _calculate_orientation_pca(patch):
    pcy, pcx = patch.shape[0] / 2.0, patch.shape[1] / 2.0

    lbls = ndi.label(patch, structure=ndi.generate_binary_structure(2, 2))[0]

    rgns = measure.regionprops(lbls)
    dists = [
        (r.label, distance.euclidean((pcy, pcx), (r.centroid[0], r.centroid[1])))
        for r in rgns
    ]
    dists = sorted(dists, key=lambda x: x[1])
    orintations = [r.orientation for r in rgns if r.label == dists[0][0]]
    centroids = [r.centroid for r in rgns if r.label == dists[0][0]]

    orientation = orintations[0]
    y0, x0 = centroids[0]

    return orientation, y0, x0


# 2D Tensor-based approach
def _calculate_orientation_tensor(patch):
    Axx, Axy, Ayy = feature.structure_tensor(patch, sigma=0.1)
    tensor_vals = np.array([[np.mean(Axx), np.mean(Axy)], [np.mean(Axy), np.mean(Ayy)]])

    w, v = np.linalg.eig(tensor_vals)
    orientation = math.atan2(*v[:, np.argmax(w)])
    y0, x0 = patch.shape[0] / 2, patch.shape[1] / 2

    return orientation, y0, x0


def estimate_2d_orientation(
    fiber_skel,
    paddding=25,
    window_radius=12,
    orient_type="tensor",
):
    """Computes orientation and diameter of fibers at every point of a skeleton.

    Estimates orientation and diameter of fibers at every point of a skeleton. The orientation
    is estimated using either tensor-based or PCA-based approach. The distance is evaluated
    by scanning a mask of fibers in a speficied angular range.

    Parameters
    ----------
    fiber_mask : 2D array
        Indicates the binary data of fibers produced by the segmentation process.

    fiber_skel : 2D array
        Indicates the skeleton produced by thinning of the binary data of fibers.

    paddding : integer
        Indicates the amount of padding at the corners to prevent an estimation error.

    window_radius : integer
        Indicates the radius of the local window of orientation calculation, which leads to
        patches of size (`window_radius`*2+1) x (`window_radius`*2+1).

    orient_type : str
        Indicates the type of algorithm for orientation estimation ('tensor' or 'pca').

    diameter_window_radius : integer
        Indicates the radius of the local window of diameter estimation, which leads to
        patches of size (`diameter_window_radius`*2+1) x (`diameter_window_radius`*2+1).

    Returns
    -------
    (clear_fiber_skel, fiber_skel, output_orientation_map, output_diameter_map,
    orientation_vals, diameter_vals) : tuple of arrays
        The skeleton with removed intersections, the skeleton, the orientation map, the
        diameter map, the arrays of orientation and diameter values.
    """
    padded_fiber_skel = np.pad(
        fiber_skel, pad_width=(paddding,), mode="constant", constant_values=0
    )

    output_orientation_map = np.zeros_like(padded_fiber_skel, dtype=np.float32)

    ycords, xcords = np.nonzero(padded_fiber_skel)

    clear_padded_fiber_skel = padded_fiber_skel.copy()

    method = feature.corner_harris(clear_padded_fiber_skel, sigma=1.5)
    corner_points = feature.corner_peaks(method, min_distance=3)

    for _, (yy, xx) in enumerate(corner_points):
        clear_padded_fiber_skel[yy - 1 : yy + 2, xx - 1 : xx + 2] = np.zeros(
            (3, 3), dtype=clear_padded_fiber_skel.dtype
        )

    for _, (yy, xx) in enumerate(zip(ycords, xcords)):
        patch_skel = padded_fiber_skel[
            (yy - window_radius) : (yy + window_radius + 1),
            (xx - window_radius) : (xx + window_radius + 1),
        ]

        orientation = None
        if orient_type == "tensor":
            orientation, _, _ = _calculate_orientation_tensor(patch_skel)
        elif orient_type == "pca":
            orientation, _, _ = _calculate_orientation_pca(patch_skel)

        final_orientation = (
            orientation if orientation >= 0.0 else (np.pi - np.abs(orientation))
        )

        output_orientation_map[yy, xx] = final_orientation

    output_orientation_map = output_orientation_map[
        paddding:-paddding, paddding:-paddding
    ]

    return output_orientation_map
