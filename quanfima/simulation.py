from __future__ import print_function
import time
import os
import itertools
import numpy as np
from multiprocessing import Pool
from scipy import ndimage as ndi
from sklearn import metrics
from skimage import filters, morphology, data as skidata, exposure, draw


def random_in(rng, number=1):
    """Returns a random value within a given range.
    """
    start, end = rng
    values = np.random.random_sample(number) * (end - start) + start
    return values[0] if number == 1 else values


def mkfiber(dims_size, length, radius, azth, lat, offset_xyz):
    """Computes fiber coordinates and its length.

    Computes a fiber of speficied `length`, `radius`, oriented under azimuth `azth` and
    latitude / elevation `lat` angles shifted to `offset_xyz` from the center of a volume of
    size `dims_size`.

    Parameters
    ----------
    dims_size : tuple
        Indicates the size of the volume.

    length : integer
        Indicates the length of the simulated fiber.

    radius : integer
        Indicates the radius of the simulated fiber.

    azth : float
        Indicates the azimuth component of the orientation angles of the fiber in radians.

    lat : float
        Indicates the latitude / elevation component of the orientation angles of the fiber
        in radians.

    offset_xyz : tuple
        Indicates the offset from the center of the volume where the fiber will be generated.

    Returns
    -------
    fiber_pts, fiber_len : tuple of array and number
        The array of fiber coordinates and the length.
    """
    dims_size = np.array(dims_size)

    half_pi = np.pi / 2.
    mx = np.array([[1., 0., 0],
                   [0., np.cos(lat), np.sin(lat)],
                   [0., -np.sin(lat), np.cos(lat)]])

    azth += half_pi
    mz = np.array([[np.cos(azth), -np.sin(azth), 0],
                   [np.sin(azth), np.cos(azth), 0],
                   [0., 0., 1.]])

    # Directional vector
    dl = 1
    dir_vec = np.array([0, 0, 1])
    dir_vec = np.dot(mx, dir_vec)
    dir_vec = np.dot(mz, dir_vec)
    dx, dy, dz = dir_vec[0], dir_vec[1], dir_vec[2]

    # Compute length
    n_steps = np.round(length / dl)
    half_steps = int(np.ceil(n_steps / 2.))
    steps = range(half_steps - int(n_steps), half_steps)

    # Draw circle perpedicular to the directional vector
    X, Y = draw.circle(0, 0, radius)
    Z = np.repeat(0, len(Y))
    circle_pts = np.array([X, Y, Z])
    circle_pts = np.dot(mx, circle_pts)
    circle_pts = np.dot(mz, circle_pts)

    # Propogate the circle profile along the directional vector
    slice_pts = np.array(zip(*circle_pts))
    dxyz = np.array([dx, dy, dz])
    step_shifts = np.array([step * dxyz for step in steps])  # [(dx,dy,dz), ...] for each step
    center_shift = dims_size * 0.5 + offset_xyz  # (x, y ,z)
    slices_pts = np.round(np.array([slice_pts + step_shift + center_shift
                                        for step_shift in step_shifts]))

    # Filter all the points which are outside the boundary
    pt_filter = lambda pt: np.all(np.greater_equal(pt, (0, 0, 0))) and \
                           np.all(np.less(np.array(pt), dims_size))
    n_slices = 0
    fiber_pts = None
    for pts in slices_pts:
        slice_pts_mask = [pt_filter(pt) for pt in pts]
        slice_pts = pts[slice_pts_mask].astype(np.int32)
        if len(slice_pts) > 0:
            n_slices += 1
            fiber_pts = slice_pts if fiber_pts is None else \
                                                np.concatenate((fiber_pts, slice_pts))
    # number of slices, e.g. steps.
    fiber_len = np.round(n_slices * dl).astype(np.int32)
    fiber_pts = fiber_pts.astype(np.int32)
    return fiber_pts, fiber_len


def simulate_fibers(volume_shape, n_fibers=1, radius_lim=(4, 10), length_lim=(0.2, 0.8),
                    lat_lim=(0, np.pi), azth_lim=(0, np.pi), gap_lim=(3, 10),
                    max_fails=10, max_len_loss=0.5, intersect=False):
    """Simulates fibers in a volume.

    Simulates `n_fibers` of the radii and lengths in ranges `radius_lim` and `length_lim`,
    oriented in a range of azimuth `azth_lim` and latitude \ elevation 'lat_lim' angles,
    separated with a gap in a range of `gap_lim`. The simulation process stops if the number
    of attempts to generate a fiber exceeds `max_fails`.

    Parameters
    ----------
    volume_shape : tuple
        Indicates the size of the volume.

    n_fibers : integer
        Indicates the number of fibers to be generated.

    radius_lim : tuple
        Indicates the range of radii for fibers to be generated.

    length_lim : tuple
        Indicates the range of lengths for fibers to be generated.

    lat_lim : tuple
        Indicates the range of latitude / elevation component of the orientation angles of
        the fibers to be generated.

    azth_lim : tuple
        Indicates the range of azimuth component of the orientation angles of the fibers to
        be generated.

    gap_lim : tuple
        Indicates the range of gaps separating the fibers from each other.

    max_fails : integer
        Indicates the maximum number of failures during the simulation process.

    max_len_loss : float
        Indicates the maximum fraction of the generated fiber placed out of volume, exceeding
        which the fiber is counted as failed.

    intersect : boolean
        Specifies if generated fibers can intersect.

    Returns
    -------
    (volume, lat_ref, azth_ref, diameter, n_generated, elapsed_time) : tuple of arrays and numbers
        The binary volume of generated fibers, the volumes of latitude / elevation and
        azimuth angles at every fiber point, the volume of diameters at every fibers point,
        the number of generated fibers and the simulation time.
    """
    ts = time.time()

    volume = np.zeros(volume_shape, dtype=np.int32)
    lat_ref = np.zeros_like(volume, dtype=np.float32)
    azth_ref = np.zeros_like(volume, dtype=np.float32)
    diameter = np.zeros_like(volume, dtype=np.float32)

    dims = np.array(volume.shape)[::-1]
    offset_lim = zip(-dims / 2, dims / 2)

    n_generated = 0
    n_fails = 0
    while n_generated < n_fibers and n_fails < max_fails:
        length = min(volume_shape)
        length = np.floor(length * random_in(length_lim, number=1)).astype(np.int32)

        offset = [random_in(olim, number=1) for olim in offset_lim]
        offset = np.round(offset).astype(np.int32)

        azth = random_in(azth_lim, number=1)
        lat = random_in(lat_lim, number=1)
        radius = random_in(radius_lim, number=1)

        gap = random_in(gap_lim, number=1)
        gap = np.round(gap).astype(np.int32)

        fiber_pts, fiber_len = mkfiber(dims, length, radius, azth, lat, offset)
        gap_fiber_pts, gap_fiber_len = mkfiber(dims, length, radius + gap, azth, lat, offset)

        # Length loss
        if (1. - float(gap_fiber_len) / length) > max_len_loss:
            n_fails = n_fails + 1
            continue

        # Intersection
        if gap_fiber_pts.size:
            X_gap, Y_gap, Z_gap = gap_fiber_pts[:, 0], gap_fiber_pts[:, 1], gap_fiber_pts[:, 2]
            X, Y, Z = fiber_pts[:, 0], fiber_pts[:, 1], fiber_pts[:, 2]

            if not intersect:
                if np.any(volume[Z_gap, Y_gap, X_gap]):
                    n_fails = n_fails + 1

                    if n_fails == max_fails:
                        print("The number of fails exceeded. Generated {} fibers".\
                                    format(n_generated))

                    continue

            # Fill the volume
            volume[Z, Y, X] = 1
            lat_ref[Z, Y, X] = lat
            azth_ref[Z, Y, X] = azth
            diameter[Z, Y, X] = radius * 2
            n_generated = n_generated + 1
            n_fails = 0

    te = time.time()
    elapsed_time = te - ts

    return (volume, lat_ref, azth_ref, diameter, n_generated, elapsed_time)


def generate_datasets(volume_size=(512, 512, 512), n_fibers=50, radius_lim=(4, 10),
                      length_lim=(0.2, 0.8), gap_lim=(3, 10), max_fails=100,
                      median_rad=3, intersect=False, output_dir=None, params=None):
    """Simulates speficied configurations of fibers and stores in a npy file.

    Simulates a number of fiber configurations speficied in `params` with `n_fibers` of the
    radii and lengths in ranges `radius_lim` and `length_lim`, separated with gaps in a range
    of `gap_lim`. The simulation process stops if the number of attempts to generate a fiber
    exceeds `max_fails`.

    Parameters
    ----------
    volume_size : tuple
        Indicates the size of the volume.

    n_fibers : integer
        Indicates the number of fibers to be generated.

    radius_lim : tuple
        Indicates the range of radii for fibers to be generated.

    length_lim : tuple
        Indicates the range of lengths for fibers to be generated.

    gap_lim : tuple
        Indicates the range of gaps separating the fibers from each other.

    max_fails : integer
        Indicates the maximum number of failures during the simulation process.

    median_rad : integer
        Indicates the radius of median filter to fill holes occured due to rounding of
        coordinates of the generated fibers.

    intersect : boolean
        Specifies if generated fibers can intersect.

    output_dir : str
        Indicates the path to the output folder where the data will be stored.

    params : dict
        Indicates the configurations of orientation of fibers to be generated.

    Returns
    -------
    out : dict
        The dictionary of generated datasets of specified configurations.
    """
    if params is None:
        params = {'aligned': {'lat_rng': (15, 15), 'azth_rng': (27, 27)},
                  'medium': {'lat_rng': (0, 45), 'azth_rng': (-45, 45)},
                  'disordered': {'lat_rng': (0, 90), 'azth_rng': (-89, 90)}}

    out = {}
    for name, config in params.items():
        data, lat_data, azth_data, diameter_data, n_gen_fibers, elapsed_time = \
                simulate_fibers(volume_size,
                                lat_lim=tuple([np.deg2rad(v) for v in config['lat_rng']]),
                                azth_lim=tuple([np.deg2rad(v) for v in config['azth_rng']]),
                                radius_lim=radius_lim,
                                n_fibers=n_fibers,
                                max_fails=max_fails,
                                gap_lim=gap_lim,
                                length_lim=length_lim,
                                intersect=intersect)

        data_8bit = data.astype(np.uint8)
        data_8bit = ndi.binary_fill_holes(data_8bit)
        data_8bit = ndi.median_filter(data_8bit, footprint=morphology.ball(median_rad))
        lat_data = ndi.median_filter(lat_data, footprint=morphology.ball(median_rad))
        azth_data = ndi.median_filter(azth_data, footprint=morphology.ball(median_rad))
        diameter_data = ndi.median_filter(diameter_data, footprint=morphology.ball(median_rad))

        out[name] = {'data': data_8bit,
                     'lat': lat_data,
                     'azth': azth_data,
                     'diameter': diameter_data,
                     'skeleton': morphology.skeletonize_3d(data_8bit).astype(np.float32),
                     'props': {'n_gen_fibers': n_gen_fibers,
                               'time': elapsed_time,
                               'intersect': intersect}}

    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.save(os.path.join(output_dir, 'dataset_fibers_n{}_r{}_{}_g{}_{}_mr{}_i{}.npy'.
                                            format(n_fibers,
                                                   radius_lim[0], radius_lim[-1],
                                                   gap_lim[0], gap_lim[-1],
                                                   median_rad, int(intersect))), out)

    return out


def simulate_particles(volume_shape, n_particles=1, radius_lim=(3, 30), max_fails=10):
    """Simulates particles in a volume.

    Simulates `n_particles` of the radii in a range `radius_lim`. The simulation process
    stops if the number of attempts to generate a particle exceeds `max_fails`.

    Parameters
    ----------
    volume_shape : tuple
        Indicates the size of the volume.

    n_particles : integer
        Indicates the number of particles to be generated.

    radius_lim : tuple
        Indicates the range of radii for particles to be generated.

    max_fails : integer
        Indicates the maximum number of failures during the simulation process.

    Returns
    -------
    (volume, diameter, n_generated, elapsed_time) : tuple of arrays and numbers
        The binary volume of generated particles, the volume of diameters at every point of
        particles, the number of generated particles and the simulation time.
    """
    ts = time.time()

    volume = np.zeros(volume_shape, dtype=np.uint8)
    diameter = np.zeros_like(volume, dtype=np.int32)

    dims = np.array(volume.shape)
    offset_lim = zip(itertools.repeat(0), dims)

    n_generated = 0
    n_fails = 0
    while n_generated < n_particles and n_fails < max_fails:
        if (n_generated % 100 == 0) or (n_generated == n_particles):
            print('n_generated = {}/{}, n_fails = {}/{}'.format(n_generated, n_particles,
                                                                n_fails, max_fails))

        offset = [random_in(olim, number=1) for olim in offset_lim]
        offset = np.round(offset).astype(np.int32)

        radius = np.round(random_in(radius_lim, number=1))

        gen_ball = morphology.ball(radius, dtype=np.int32)
        Z, Y, X = gen_ball.nonzero()

        Z += offset[0]
        Y += offset[1]
        X += offset[2]

        if np.max(X) >= dims[0] or np.min(X) < 0 or \
           np.max(Y) >= dims[1] or np.min(Y) < 0 or \
           np.max(Z) >= dims[2] or np.min(Z) < 0:

            n_fails = n_fails + 1
            continue

        if np.any(volume[Z, Y, X]):
            n_fails = n_fails + 1

            if n_fails == max_fails:
                print("The number of fails exceeded. Generated {} particles".\
                            format(n_generated))

            continue

        # Fill the volume
        volume[Z, Y, X] = 1
        diameter[Z, Y, X] = radius * 2
        n_generated = n_generated + 1
        n_fails = 0

    te = time.time()
    elapsed_time = te - ts

    return (volume, diameter, n_generated, elapsed_time)


def generate_particle_dataset(volume_size=(512, 512, 512), n_particles=500,
                              radius_lim=(4, 10), max_fails=100, output_dir=None):
    """Simulates a speficied number of particles and stores complete dataset in a npy file.

    Parameters
    ----------
    volume_size : tuple
        Indicates the size of the volume.

    n_particles : integer
        Indicates the number of particles to be generated.

    radius_lim : tuple
        Indicates the range of radii for particles to be generated.

    max_fails : integer
        Indicates the maximum number of failures during the simulation process.

    output_dir : str
        Indicates the path to the output folder where the data will be stored.

    Returns
    -------
    out : dict
        The dictionary of generated dataset.
    """
    out = {}

    data, diameter_data, n_gen_particle, elapsed_time = \
                            simulate_particles(volume_size,
                                               n_particles=n_particles,
                                               radius_lim=radius_lim,
                                               max_fails=max_fails)

    out['normal'] = {'data': data,
                     'diameter': diameter_data,
                     'props': {'n_gen_particles': n_gen_particle,
                               'time': elapsed_time}}

    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.save(os.path.join(output_dir,
                         'dataset_particles_n{}_r{}_{}.npy'.format(n_particles,
                                                                   radius_lim[0],
                                                                   radius_lim[-1])), out)
    return out


def generate_blobs(volume_size, blob_size_fraction=0.1, transparency_ratio=0.5, sigma=90.):
    """Generates random blobs smoothed with Gaussian filter in a volume.

    Generates several blobs of random size in a volume using function from scikit-image,
    which are subsequently smoothed with a Gaussian filter of a large sigma to imitate
    3D uneven illumination of the volume.

    Parameters
    ----------
    volume_size : tuple
        Indicates the size of the volume.

    blob_size_fraction : float
        Indicates the fraction of volume occupied by blobs.

    transparency_ratio : float
        Indicates the transparency of blobs in a range [0, 1].

    sigma : float
        Indicates the sigma of Gaussian filter.

    Returns
    -------
    blobs_smeared : ndarray
        The volume with smoothed blobs of a specified transparency.
    """
    blobs = skidata.binary_blobs(length=max(volume_size),
                                 blob_size_fraction=blob_size_fraction,
                                 n_dim=len(volume_size),
                                 seed=1)
    blobs = blobs.astype(np.float32)

    blobs_smeared = ndi.gaussian_filter(blobs, sigma) * transparency_ratio
    return blobs_smeared


def generate_noised_data(datasets_path, noise_levels=[0.0, 0.15, 0.3],
                         smooth_levels=[0.0, 1.0, 2.0], blobs=None, use_median=True,
                         median_rad=3, output_dir=None, n_processes=9):
    """Contaminates datasets with a speficied additive Gaussian noise and smoothing level.

    Contaminates the datasets (generated with `generate_datasets` function) with the specified
    level of additive Gaussian noise and smoothing, uneven illumination can be added if
    `blobs` is provided. The contaminating process can be performed in a parallel `n_processes`
    processes.

    Parameters
    ----------
    datasets_path : str
        Indicates the path to dataset.

    noise_levels : array
        Indicates the array of standard deviations of noise.

    smooth_levels : array
        Indicates the array of sigma values of Gaussian filter.

    blobs : ndarray
        Indicates the volume of uneven illumination generated by `generate_blobs`.

    use_median : boolean
        Specifies if the median filter should be applied after addition of noise.

    median_rad : integer
        Indicates the size of median filter.

    output_dir : str
        Indicates the path to the output folder where the data will be stored.

    n_processes : integer
        Indicates the number of parallel processes.

    Returns
    -------
    results : array of dicts
        The array of dictionaries containing paths to contaminated datasets, and other
        properties.
    """
    datasets_names = np.load(datasets_path).item().keys()
    n_datasets = len(datasets_names)

    dataset_filename = os.path.basename(datasets_path)
    dataset_filename = os.path.splitext(dataset_filename)[0]
    output_dir = os.path.join(output_dir, dataset_filename)

    data_items = [(dname, dpath, blb, odir)
                      for dname, dpath, blb, odir in zip(datasets_names,
                                                         [datasets_path]*n_datasets,
                                                         [blobs]*n_datasets,
                                                         [output_dir]*n_datasets)]

    params = [data_items, noise_levels, smooth_levels]
    params = [p for p in itertools.product(*params)]

    proc_pool = Pool(processes=n_processes)
    results = proc_pool.map(unpack_additive_noise, params)
    proc_pool.close()
    proc_pool.join()
    proc_pool.terminate()

    np.save(os.path.join(output_dir, 'params.npy'), results)

    return results


def unpack_additive_noise(args):
    """Unpack arguments and return result of `additive_noise` function.
    """
    return additive_noise(*args)


def additive_noise(params, noise_lvl, smooth_lvl, use_median=True, median_rad=3):
    """Contaminates datasets with a speficied additive Gaussian noise and smoothing level.

    Contaminates the datasets (generated with `generate_datasets` function) with the specified
    level of additive Gaussian noise and smoothing, uneven illumination can be added by
    extracting `blobs` from `params` tuple with some other arguments.

    Parameters
    ----------
    params : tuple
        Contains `name`, `dataset_path`, `blobs`, `output_dir` arguments passed from
        `generate_noised_data`.

    noise_level : float
        Indicates the standard deviations of noise.

    smooth_level : float
        Indicates the sigma value of Gaussian filter.

    use_median : boolean
        Specifies if the median filter should be applied after addition of noise.

    median_rad : integer
        Indicates the size of median filter.

    Returns
    -------
    datasets_props : dict
        The dictionary containing the path to the reference dataset, the path to the
        contaminated dataset, the generated name, the SNR level, the precision, the recall
        and f1-score, and the level of noise and smoothing.
    """
    name, dataset_path, blobs, output_dir = params

    datasets = np.load(dataset_path).item()
    data = datasets[name]['data']
    data_skel = datasets[name]['skeleton']

    def median_fltr(data, footprint):
        out = np.zeros_like(data, dtype=np.uint8)

        for i in xrange(data.shape[0]):
            out[i] = filters.rank.median(data[i], selem=footprint)

        return out

    def threshold_dataset(data):
        data_seg = np.zeros_like(data, dtype=np.uint8)
        data_8bit = exposure.rescale_intensity(data, in_range='image',
                                               out_range=np.uint8).astype(np.uint8)

        for i in xrange(data_seg.shape[0]):
            dslice = data_8bit[i]
            th_val = filters.threshold_otsu(dslice)
            data_seg[i] = (dslice > th_val).astype(np.uint8)

        return data_seg

    print('{}: Noise: {} | Smooth: {}'.format(name, noise_lvl, smooth_lvl))

    data_ref = data.astype(np.float32)
    data_ref_skel = exposure.rescale_intensity(data_skel,
                                               out_range=(0, 1)).astype(np.uint8)

    data_noised = data_ref + noise_lvl * np.random.randn(*data_ref.shape)

    if (blobs is not None) and (noise_lvl != 0.) and (smooth_lvl != 0.):
        data_noised += blobs

    if smooth_lvl != 0:
        data_noised = ndi.gaussian_filter(data_noised, smooth_lvl)

    snr = np.mean(data_noised[data_ref != 0]) / np.std(data_noised[data_ref == 0])
    data_noised = exposure.rescale_intensity(data_noised, out_range=np.uint8).astype(np.uint8)

    if use_median and (noise_lvl != 0.) and (smooth_lvl != 0.):
        data_noised = median_fltr(data_noised, morphology.disk(median_rad))

    data_noised_seg = threshold_dataset(data_noised)
    data_noised_skel = morphology.skeletonize_3d(data_noised_seg)

    precision, recall, fbeta_score, support = \
                    metrics.precision_recall_fscore_support(data_ref_skel.flatten(),
                                                            data_noised_skel.flatten(),
                                                            beta=1.0,
                                                            pos_label=1,
                                                            average='binary')

    print('Precision: {}, Recall: {}, F1-score: {}'.format(precision, recall, fbeta_score))

    data_out = {'data': data_ref,
                'data_noised': data_noised,
                'skeleton': data_ref_skel,
                'skeleton_noised': data_noised_skel,
                'seg_noised': data_noised_seg}

    dataset_outpath = os.path.join(output_dir,
                                'dataset_noised_fibers_{}_nl{}_sl{}.npy'.
                                            format(name, noise_lvl, smooth_lvl))

    datasets_props = {'ref_dataset_path': dataset_path,
                      'dataset_path': dataset_outpath,
                      'name': name,
                      'snr': snr,
                      'precision': precision,
                      'recall': recall,
                      'f1_score': fbeta_score,
                      'adgauss_std': noise_lvl,
                      'smooth_sigma': smooth_lvl}

    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        np.save(dataset_outpath, data_out)

    return datasets_props
