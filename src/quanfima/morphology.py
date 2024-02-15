import numpy as np
from skimage import measure
from scipy import ndimage as ndi
import pandas as pd

# Constants of object counter
_MEASUREMENTS = {"Label": "label", "Area": "area", "Perimeter": "perimeter"}


def _calc_sphericity(volume, perimeter):
    """Computes sphericity from volume and perimeter of an object."""
    r = ((3.0 * volume) / (4.0 * np.pi)) ** (1.0 / 3.0)
    return (4.0 * np.pi * (r * r)) / perimeter


def object_counter(stack_binary_data):
    """Label and counts particles in a binary data.

    Parameters
    ----------
    stack_binary_data : 3D array
        Indicates the 3D binary data.

    Returns
    -------
    (objects_stats, labeled_stack) : tuple
        The tuple of a DataFrame object of counted partilces and the labeled 3D data.
    """
    measurements_vals = _MEASUREMENTS.values()

    print("Object counting - Labeling...")
    labeled_stack, num_labels = ndi.measurements.label(stack_binary_data)
    objects_stats = pd.DataFrame(columns=measurements_vals)

    print("Object counting - Stats gathering...")
    for slice_idx in np.arange(labeled_stack.shape[0]):
        for region in measure.regionprops(labeled_stack[slice_idx]):
            objects_stats = objects_stats.append(
                {mname: region[mval] for mname, mval in _MEASUREMENTS.items()},
                ignore_index=True,
            )

    print("Object counting - Stats grouping...")
    objects_stats = objects_stats.groupby("Label", as_index=False).sum()
    objects_stats["Sphericity"] = _calc_sphericity(
        objects_stats["Area"], objects_stats["Perimeter"]
    )

    return (objects_stats, labeled_stack)


def calc_porosity(data):
    """Computes porosity.

    Parameters
    ----------
    data : 3D array
        Indicates the labeled 3D data.

    Returns
    -------
    out : dict
        The dictionary of materials and corresponding porosity values.
    """
    total_volume = data.size
    mats = np.unique(data)
    out = {
        "Material {}".format(m): 1.0 - (float(data[data == m].size) / total_volume)
        for m in mats[mats > 0]
    }
    return out
