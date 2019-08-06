from __future__ import print_function
import math
import time
import os
import itertools
import numpy as np
from skimage import feature, measure, filters
from skimage.util.shape import view_as_blocks
from scipy import ndimage as ndi
from scipy.spatial import distance
import vigra
import pandas as pd
from multiprocessing import Pool
from quanfima import cuda_available

if cuda_available:
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    from pycuda import gpuarray

# Constants of object counter
_MEASUREMENTS = {
    'Label': 'label',
    'Area': 'area',
    'Perimeter': 'perimeter'
}


def cast_ray(theta, y0, x0, fiber_mask, ray_len=100):
    """Computes a distance between two detected points at `fiber_mask`.

    Casts a ray from the point (x0, y0) at `fiber_mask` under angle `theta` of maximum
    length of `ray_len`. The ray is casted towards to opposite directions, thus two points
    (x1,y1) and (x2,y2) are detected at opposite borders of fiber. Finally, the Euclidean
    distance between these points is calculated and returned.

    Parameters
    ----------
    theta : float
        Indicates the angle under which the ray is casted.

    y0 : integer
        Indicates the y-axis component of the origin of ray emission.

    x0 : integer
        Indicates the x-axis component of the origin of ray emission.

    fiber_mask : 2D array
        Indicates the binary array of fibers.

    ray_len : integer
        Indicates the maximum length of the ray.

    Returns
    -------
    distance : float
        The distance between (x1,y0) and (x2,y2) detected points.
    """
    m = math.tan(theta)
    y1, x1, y2, x2 = 0, 0, 0, 0

    for i in range(-ray_len, 0)[::-1]:
        if np.abs(m) <= 1.:
            y = int(y0 - i * m)
            x = int(x0 + i)
        else:
            y = int(y0 - i)
            x = int(x0 + i * 1./m)

        if x < 0 or x >= fiber_mask.shape[1] or \
           y < 0 or y >= fiber_mask.shape[0]:
                continue

        if fiber_mask[y, x] == 0:
            break

        y1, x1 = y, x

    for i in range(ray_len):
        if np.abs(m) <= 1.:
            y = int(y0 - i * m)
            x = int(x0 + i)
        else:
            y = int(y0 - i)
            x = int(x0 + i * 1./m)

        if x < 0 or x >= fiber_mask.shape[1] or \
           y < 0 or y >= fiber_mask.shape[0]:
                continue

        if fiber_mask[y, x] == 0:
            break

        y2, x2 = y, x

    return distance.euclidean((y1, x1), (y2, x2))


def scan_fiber_thickness(angle, patch, angular_step=1., tilt_range=range(-5, 6), ray_len=100):
    """Computes an average diameter of a structure within a patch.

    Estimates an average distance from sequence of distances calculated with `cast_ray`
    function for a range of angles `tilt_range`.

    Parameters
    ----------
    orientation : float
        Indicates the orientation angle of a structure centered within a patch.

    patch : 2D array
        Indicates the patch with a centered structure having orientation `angle`.

    angular_step : float
        Indicates the angular step of the scanning range in degrees.

    tilt_range : integer
        Indicates the scanning range of steps to estimate the distance.

    ray_len : integer
        Indicates the maximum length of the ray.

    Returns
    -------
    average_diameter : float
        The average diameter of a structure centered within a patch.
    """
    thickness_dist = []

    y0, x0 = patch.shape[0]/2, patch.shape[1]/2
    pi2 = np.pi/2.
    rad_p_deg = np.deg2rad(angular_step)

    for tilt_amount in tilt_range:
        theta = angle + pi2 + tilt_amount * rad_p_deg
        thickness_dist.append(cast_ray(theta, y0, x0, patch, ray_len=ray_len))

    average_distance = np.mean(thickness_dist)
    return average_distance


def estimate_fiber_properties(fiber_mask, fiber_skel, paddding=25, window_radius=12,
                              orient_type='tensor', diameter_window_radius=12):
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
    orientation_vals, diameter_vals = [], []

    padded_fiber_skel = np.pad(fiber_skel, pad_width=(paddding,), mode='constant', constant_values=0)
    padded_fiber_mask = np.pad(fiber_mask, pad_width=(paddding,), mode='constant', constant_values=0)

    output_orientation_map, output_diameter_map = \
                            np.zeros_like(padded_fiber_skel, dtype=np.float32), \
                            np.zeros_like(padded_fiber_skel, dtype=np.float32)
    
    pcy, pcx = (window_radius*2+1)/2., (window_radius*2+1)/2.

    ycords, xcords = np.nonzero(padded_fiber_skel)

    clear_padded_fiber_skel = padded_fiber_skel.copy()

    method = feature.corner_harris(clear_padded_fiber_skel, sigma=1.5)
    corner_points = feature.corner_peaks(method, min_distance=3)

    for i, (yy, xx) in enumerate(corner_points):
        clear_padded_fiber_skel[yy-1:yy+2, xx-1:xx+2] = np.zeros((3, 3),
                                            dtype=clear_padded_fiber_skel.dtype)

    # PCA-based approach
    def calc_orientation_pca(patch):
        lbls = ndi.label(patch, structure=ndi.generate_binary_structure(2, 2))[0]

        rgns = measure.regionprops(lbls)
        dists = [(r.label, distance.euclidean((pcy, pcx), (r.centroid[0], r.centroid[1]))) for r in rgns]
        dists = sorted(dists, key=lambda x: x[1])
        orintations = [r.orientation for r in rgns if r.label == dists[0][0]]
        centroids = [r.centroid for r in rgns if r.label == dists[0][0]]

        orientation = orintations[0]
        y0, x0 = centroids[0]

        return orientation, y0, x0

    # Tensor-based approach
    def calc_orientation_tensor(patch):
        Axx, Axy, Ayy = feature.structure_tensor(patch, sigma=0.1)
        tensor_vals = np.array([[np.mean(Axx), np.mean(Axy)], [np.mean(Axy), np.mean(Ayy)]])

        w, v = np.linalg.eig(tensor_vals)
        orientation = math.atan2(*v[:,np.argmax(w)])
        y0, x0 = patch.shape[0]/2, patch.shape[1]/2

        return orientation, y0, x0

    for i, (yy, xx) in enumerate(zip(ycords, xcords)):
        patch_skel = padded_fiber_skel[(yy-window_radius):(yy+window_radius+1),
                                       (xx-window_radius):(xx+window_radius+1)]

        patch_mask = padded_fiber_mask[(yy-window_radius):(yy+window_radius+1),
                                       (xx-window_radius):(xx+window_radius+1)]

        if orient_type == 'tensor':
            orientation, y0, x0 = calc_orientation_tensor(patch_skel)
        elif orient_type == 'pca':
            orientation, y0, x0 = calc_orientation_pca(patch_skel)

        final_orientation = orientation if orientation >= 0.0 else (np.pi - np.abs(orientation))
        orientation_vals.append(final_orientation)
        output_orientation_map[yy, xx] = final_orientation

        if clear_padded_fiber_skel[yy, xx] != 0:
            gy, gx = yy - window_radius + y0, xx - window_radius + x0

            fiber_tpatch = padded_fiber_mask[yy-diameter_window_radius:yy+diameter_window_radius+1,
                                             xx-diameter_window_radius:xx+diameter_window_radius+1]

            final_thickness = scan_fiber_thickness(orientation, patch_mask)
            output_diameter_map[yy, xx] = final_thickness
            diameter_vals.append(final_thickness)

    output_orientation_map = output_orientation_map[paddding:-paddding, paddding:-paddding]
    output_diameter_map = output_diameter_map[paddding:-paddding, paddding:-paddding]
    clear_fiber_skel = clear_padded_fiber_skel[paddding:-paddding, paddding:-paddding]

    return (clear_fiber_skel, fiber_skel,
            output_orientation_map, output_diameter_map,
            orientation_vals, diameter_vals)

def estimate_fourier_orientation(data, grid_shape=(2,2), sigma=2., zoom=1., order=3):
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
    block_shape = tuple([int(math.floor(d/float(gs))) for d, gs in
                                                zip(data.shape, grid_shape)])
    data_blocks = view_as_blocks(data, block_shape=block_shape)
    orient_blocks = np.zeros(grid_shape, dtype=np.float32)

    for i in xrange(grid_shape[0]):
        for j in xrange(grid_shape[1]):
            y0, x0 = block_shape[0] * j + block_shape[0]/2, \
                     block_shape[1] * i + block_shape[1]/2
            dblock = data_blocks[i, j]

            if zoom != 1.:
                dblock = ndi.interpolation.zoom(dblock, zoom, order=order)

            dblock_freq = np.abs(np.fft.fftshift(np.fft.fft2(dblock)).real)
            dblock_freq = ndi.gaussian_filter(dblock_freq, sigma=sigma)
            dblock_mask = (dblock_freq > filters.threshold_otsu(dblock_freq)).astype(np.uint8)

            lbls = ndi.label(dblock_mask,
                             structure=ndi.generate_binary_structure(2,2))[0]
            rgns = measure.regionprops(lbls)

            dists = [(r.label, r.area) for r in rgns]
            dists = sorted(dists, key=lambda x: x[1])

            orient_blocks[i, j] = [r.orientation for r in rgns
                                                if r.label == dists[0][0]][0]

    return (orient_blocks, block_shape)


def numpy3d_to_array(np_array, allow_surface_bind=True):
    """Converts 3D numpy array to 3D device array.
    """
    # numpy3d_to_array
    # taken from pycuda mailing list (striped for C ordering only)

    d, h, w = np_array.shape

    descr = cuda.ArrayDescriptor3D()
    descr.width = w
    descr.height = h
    descr.depth = d
    descr.format = cuda.dtype_to_array_format(np_array.dtype)
    descr.num_channels = 1
    descr.flags = 0

    if allow_surface_bind:
        descr.flags = cuda.array3d_flags.SURFACE_LDST

    device_array = cuda.Array(descr)

    copy = cuda.Memcpy3D()
    copy.set_src_host(np_array)
    copy.set_dst_array(device_array)
    copy.width_in_bytes = copy.src_pitch = np_array.strides[1]
    copy.src_height = copy.height = h
    copy.depth = d

    copy()

    return device_array


def _filter_coords(data, ws, th_nval=0.8):
    """Filters coordinates of 3D local windows by number of non-zero values speficied by `th_nval`.
    """
    ws2 = ws / 2

    def check_point(data, pt, ws2, th):
        size = max(data.shape)
        lim0 = pt - ws2
        lim1 = pt + ws2

        if any(np.array(lim0) < 0) or any(np.array(lim1) > (size - 1)):
            return False

        z0, y0, x0 = lim0
        z1, y1, x1 = lim1

        vol = data[z0:z1, y0:y1, x0:x1]
        nz_vals = np.count_nonzero(vol)

        if np.count_nonzero(vol) < ws2*2.*th:
            return False

        return True

    filtered_coords = [pt for pt in np.transpose(np.nonzero(data)) if check_point(data, pt, ws2, th_nval)]

    return np.transpose(filtered_coords)


def extract_patch(data, pt, ws2):
    """Extract a path from `data` of radius `ws2`.

    Parameters
    ----------
    data : 3D array
        Indicates the 3D data array.

    pt : array or tuple
        Indicates the coordinates of a point within `data`.

    ws2 : integer
        Indicates a half of the window size of the patch to be extracted.

    Returns
    -------
    patch : 3D array
        The 3D patch extracted from the local window around the point `pt` in the `data`
        with radius of `ws2`.
    """
    lim0 = pt - ws2
    lim1 = pt + ws2

    if any(np.array(lim0) < 0) or any(np.array(lim1) > (data.shape[0] - 1)):
        return None

    z0, y0, x0 = lim0
    z1, y1, x1 = lim1
    patch = data[z0:z1, y0:y1, x0:x1]
    return patch


def orientation_3d_tensor_vigra(data, sigma=0.1):
    """Computes 3D orientation from a 3D structure tensor of `data`.

    Parameters
    ----------
    data : ndarray
        Indicates the N-dimensional array.

    sigma : float
        Indicates the sigma value of the Gaussian filter.

    Returns
    -------
    (lat, azth) : tuple of floats
        The latitude / elevation and azimuth component of 3D orientation of structures within
        the patch `data`.
    """
    img = vigra.filters.structureTensor(data, 1, 1, sigma_d=sigma)
    Axx = img[:, :, :, 0]
    Axy = img[:, :, :, 1]
    Axz = img[:, :, :, 2]
    Ayy = img[:, :, :, 3]
    Ayz = img[:, :, :, 4]
    Azz = img[:, :, :, 5]

    tensor_vals = np.array([[np.mean(Azz), np.mean(Ayz), np.mean(Axz)],
                            [np.mean(Ayz), np.mean(Ayy), np.mean(Axy)],
                            [np.mean(Axz), np.mean(Axy), np.mean(Axx)]])

    tensor_vals = tensor_vals[::-1, ::-1]

    eps = 1e-8
    w, v = np.linalg.eig(tensor_vals)

    mv = v[:, np.argmin(w)]  # z, y, x
    mv[np.abs(mv) < eps] = 0

    G = np.sqrt(mv[2]**2 + mv[1]**2)
    lat = np.arcsin(np.around(G, decimals=3))
    azth = np.arctan(mv[1] / mv[2]) if mv[2] else np.pi/2.

    return (lat, azth)


def estimate_tensor(name, skel, data, window_size, output_dir, sigma=0.025, make_output=True):
    """Computes 3D orientation at every point of a skeleton of data.

    Estimates 3D orientation at every point of the skeleton `skel` extracted from the binary
    data `data` within a 3D local window of size `window_size` using the tensor-based approach
    with a Gaussian smoothing of `sigma`.

    Parameters
    ----------
    name : str
        Indicates the name of the output npy file.

    skel : 3D array
        Indicates the skeleton of the binary data.

    data : 3D array
        Indicates the 3D binary data.

    window_size : integer
        Indicates the size of the 3D local window.

    output_dir : str
        Indicates the path to the output folder where the data will be stored.

    output_fmt : str
        Indicates the format of

    sigma : float
        Indicates the sigma value of the Gaussian filter.

    make_output : boolean
        Specifies if the estimated data should be stored.

    Returns
    -------
    output_props : dict
        The dictionary of properties specifying the sample name, the algorithm name, the number
        of processes, and the execution time.
    """
    output_props = dict()

    Z, Y, X = skel.nonzero()
    tens_lat_arr = np.zeros_like(skel, dtype=np.float32)
    tens_azth_arr = np.zeros_like(skel, dtype=np.float32)
    skel_est = np.zeros_like(skel, dtype=np.int32)

    ws = np.uint32(window_size)
    ws2 = ws/2
    skel_shape = skel.shape

    output_props['sample_name'] = name
    output_props['type'] = 'tensor'
    output_props['n_processes'] = 1

    ts = time.time()

    for idx, pt in enumerate(zip(Z, Y, X)):
        lim0 = pt - ws2
        lim1 = pt + ws2

        if any(np.array(lim0) < 0) or any(np.array(lim1) > (skel_shape[0] - 1)):
            skel_est[pt] = -1
            continue

        z0, y0, x0 = lim0
        z1, y1, x1 = lim1

        area = data[z0:z1, y0:y1, x0:x1]

        lat, azth = orientation_3d_tensor_vigra(area, sigma)

        tens_lat_arr[pt] = lat
        tens_azth_arr[pt] = azth
        skel_est[pt] = 255

    te = time.time()
    output_props['time'] = te-ts

    print("Tensor time: {}s" % (output_props['time']))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if make_output:
        opath = os.path.join(output_dir, '{}.npy').format(name)
        output_props['output_path'] = opath
        output = {'lat': tens_lat_arr, 'azth': tens_azth_arr, 'skeleton': skel_est, 'indices': skel_est > 0}
        output['props'] = output_props
        np.save(opath, output)

    return output_props


def execute_tensor(patch, sigma):
    """Executes the tensor-based approach for a `patch` and `sigma` for Gaussian smoothing.

    Parameters
    ----------
    patch : 3D aray
        Indicates the data patch of some size.

    sigma : float
        Indicates the sigma value of the Gaussian filter.

    Returns
    -------
    (lat, azth, 255) : tuple
        The latitude / elevation and azimuth components of 3D orientation with valid value
        for the point at the skeleton. If the patch is None, then return (0, 0, -1) where
        -1 indicates that this point of the skeleton is invalid.
    """
    if patch is None:
        return (0, 0, -1)

    lat, azth = orientation_3d_tensor_vigra(patch, sigma)

    return (lat, azth, 255)


def unpack_execute_tensor(args):
    """Unpack input arguments and return result of `execute_tensor` function
    """
    return execute_tensor(*args)


def estimate_tensor_parallel(name, skel, data, window_size, output_dir, sigma=0.025,
                             make_output=True, n_processes=4):
    """Computes 3D orientation at every point of a skeleton of data in parallel processes.

    Estimates 3D orientation at every point of the skeleton `skel` extracted from the binary
    data `data` within a 3D local window of size `window_size` using the tensor-based approach
    with a Gaussian smoothing of `sigma`. The orientation is estimated simultaneously at
    `n_processes` parallel processes.

    Parameters
    ----------
    name : str
        Indicates the name of the output npy file.

    skel : 3D array
        Indicates the skeleton of the binary data.

    data : 3D array
        Indicates the 3D binary data.

    window_size : integer
        Indicates the size of the 3D local window.

    output_dir : str
        Indicates the path to the output folder where the data will be stored.

    sigma : float
        Indicates the sigma value of the Gaussian filter.

    make_output : boolean
        Specifies if the estimated data should be stored.

    n_processes : integer
        Indicates the number of the parallel processes.

    Returns
    -------
    output_props : dict
        The dictionary of properties specifying the sample name, the algorithm name, the number
        of processes, and the execution time.
    """
    output_props = dict()

    Z, Y, X = skel.nonzero()
    tens_lat_arr = np.zeros_like(skel, dtype=np.float32)
    tens_azth_arr = np.zeros_like(skel, dtype=np.float32)
    skel_est = np.zeros_like(skel, dtype=np.int32)

    ws = np.uint32(window_size)
    ws2 = ws/2

    output_props['sample_name'] = name
    output_props['type'] = 'tensor_parallel'
    output_props['n_processes'] = n_processes

    ts = time.time()

    pts = zip(Z, Y, X)
    data_patches = [extract_patch(data, pt, ws2) for pt in pts]
    print(len(data_patches))
    args = zip(data_patches, itertools.repeat(sigma))

    proc_pool = Pool(processes=n_processes)
    results = np.array(proc_pool.map(unpack_execute_tensor, args))
    proc_pool.close()
    proc_pool.join()
    proc_pool.terminate()

    te = time.time()
    output_props['time'] = te-ts

    lat_arr, azth_arr, skel_arr = results.T
    tens_lat_arr[Z, Y, X] = lat_arr
    tens_azth_arr[Z, Y, X] = azth_arr
    skel_est[Z, Y, X] = skel_arr

    print("Tensor parallel time: %fs" % (output_props['time']))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if make_output:
        opath = os.path.join(output_dir, '{}.npy').format(name)
        output_props['output_path'] = opath
        output = {'lat': tens_lat_arr, 'azth': tens_azth_arr, 'skeleton': skel_est, 'indices': skel_est > 0}
        output['props'] = output_props
        np.save(opath, output)

    return output_props


def _diameter_kernel():
    """Returns the CUDA kernel to estimate 3D diameter of structures in a 3D local window.
    """
    diameter_kernel_src = """

    texture<float, cudaTextureType3D, cudaReadModeElementType> tex_data;

    __global__ void diameter3d (unsigned int width,
                                unsigned int height,
                                unsigned int depth,
                                const int n_points,
                                const float norm_factor,
                                const int max_iters,
                                const int n_scan_angles,
                                const int *X,
                                const int *Y,
                                const int *Z,
                                const float *scan_angl_arr,
                                const float *azth_data,
                                const float *lat_data,
                                float *radius_arr)
    {
        unsigned long blockId, idx;
        blockId = blockIdx.x + blockIdx.y * gridDim.x;
        idx = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

        if (idx > n_points) {
            return;
        }

        float _x, _y, _z;
        _x = (float)X[idx] + 0.5;
        _y = (float)Y[idx] + 0.5;
        _z = (float)Z[idx] + 0.5;

        // -----------------------------------------
        // Find the diameter

        float azth = azth_data[idx];
        azth += M_PI_2;
        float lat = lat_data[idx];

        float cy = cosf(lat), sy = sinf(lat);
        float cz = cosf(azth), sz = sinf(azth);
        //float cz = -sinf(azth), sz = cosf(azth); // taking into account azth + pi/2

        //vector along fiber
        //float dx = -sy * sz;
        //float dy = cy* sz;
        //float dz = cy;
        float uvec[3] = {0.0, 0.0, 1.0};
        float fiber_vector_x[3] = {0.0,
                                   uvec[2]*cy + uvec[3]*sy,
                                  -uvec[2]*sy + uvec[3]*cy};

        float fiber_vector_z[3] = {fiber_vector_x[0]*cz - fiber_vector_x[1]*sz,
                                   fiber_vector_x[0]*sz + fiber_vector_x[1]*cz,
                                   fiber_vector_x[2]};

        float dx = fiber_vector_z[0], dy = fiber_vector_z[1], dz = fiber_vector_z[2];

        //scan vector perpendicular to a fiber vector (rotation X -> Z)
        float scan_vec[3] = {0, 1, 0}; // unit vector perpendicular to default (0,0,1) direction
        float rot_scan_vec_x[3] = {0.0,
                                   scan_vec[1]*cy + scan_vec[2]*sy,
                                  -scan_vec[1]*sy + scan_vec[2]*cy};


        float rot_scan_vec_z[3] = {rot_scan_vec_x[0]*cz - rot_scan_vec_x[1]*sz,
                                   rot_scan_vec_x[0]*sz + rot_scan_vec_x[1]*cz,
                                   rot_scan_vec_x[2]};

        float out_radius = 0;

        for (int scan_angl_idx = 0; scan_angl_idx < n_scan_angles; scan_angl_idx++) {
            float theta = scan_angl_arr[scan_angl_idx];

            float ct = cosf(theta), st = sinf(theta);
            float x = rot_scan_vec_z[0], y = rot_scan_vec_z[1], z = rot_scan_vec_z[2];
            float u = dx, v = dy, w = dz;

            //rotation of point (x,y,z) around axis (u,v,w)
            float scan_vec_coords[3] =
                        {u*(u*x + v*y + w*z)*(1.0f - ct) + x*ct + (-w*y + v*z)*st,
                         v*(u*x + v*y + w*z)*(1.0f - ct) + y*ct + (w*x - u*z)*st,
                         w*(u*x + v*y + w*z)*(1.0f - ct) + z*ct + (-v*x + u*y)*st};

            float nc[3] = {_x, _y, _z};
            float p[3];

            for (int i = 0; i < max_iters; i++) {
                nc[0] += scan_vec_coords[0];
                nc[1] += scan_vec_coords[1];
                nc[2] += scan_vec_coords[2];

                if (tex3D(tex_data, nc[0], nc[1], nc[2]) == 0) {
                        p[0] = nc[0];
                        p[1] = nc[1];
                        p[2] = nc[2];
                        break;
                }
            }

            out_radius += norm3df(p[0] - _x, p[1] - _y, p[2] - _z);
        }

        radius_arr[idx] = out_radius * norm_factor;
    }
    """

    dm_program = SourceModule(diameter_kernel_src)
    diameter3d = dm_program.get_function("diameter3d")

    return dm_program, diameter3d


def estimate_diameter_gpu(skel, data, lat_data, azth_data, n_scan_angles, max_iters=150, do_reshape=True):
    """Computes 3D diameter at every point of a skeleton of data using GPU.

    Estimates 3D diameter at every point of the skeleton `skel` extracted from the binary
    data `data` with help of orientation information provided by `lat_data` and `azth_data`
    arrays. The diameter is evaluated with a ray casting approach `cast_ray` adapted for
    a 3D case.

    Parameters
    ----------
    skel : 3D array
        Indicates the skeleton of the binary data.

    data : 3D array
        Indicates the 3D binary data.

    lat_data : 3D array
        Indicates the 3D array containing latitude / elevation angle at every point of
        the skeleton in radians.

    azth_data : 3D array
        Indicates the 3D array containing azimuth angle at every point of the skeleton
        in radians.

    n_scan_angles : int
        Indicates the number of scanning angles on a range [0, 360] degrees.

    max_iters : int
        Indicates the maximum length of a ray in each direction.

    do_reshape : boolean
        Specifies if the output array should be reshaped immediately after estimation.

    Returns
    -------
    out : dict
        The dictionary of the 3D array of estimated diameter and the execution time.
    """
    if not cuda_available:
        print('The pycuda package is not found. The diameter estimation cannot be done.')
        return None

    program, diameter3d = _diameter_kernel()

    Z, Y, X = np.int32(skel.nonzero())
    depth, height, width = np.uint32(skel.shape)

    scan_angl_arr = np.deg2rad(np.float32(np.linspace(0, 360, num=n_scan_angles, endpoint=False)))
    radius_arr = np.zeros_like(Z, dtype=np.float32)

    lat_data_1d = lat_data[skel.nonzero()]
    azth_data_1d = azth_data[skel.nonzero()]

    gpu_X = gpuarray.to_gpu(X)
    gpu_Y = gpuarray.to_gpu(Y)
    gpu_Z = gpuarray.to_gpu(Z)

    gpu_lat_data_1d = gpuarray.to_gpu(lat_data_1d)
    gpu_azth_data_1d = gpuarray.to_gpu(azth_data_1d)

    gpu_radius_arr = gpuarray.to_gpu(radius_arr)
    gpu_scan_angl_arr = gpuarray.to_gpu(scan_angl_arr)

    gpu_rad_tex = program.get_texref('tex_data')
    gpu_data = numpy3d_to_array(data)
    gpu_rad_tex.set_array(gpu_data)

    n_scan_angles = np.uint32(n_scan_angles)
    n_points = np.uint32(len(Z))
    max_iters = np.uint32(max_iters)
    norm_factor = np.float32(1. / n_scan_angles)

    block = (16, 16, 1)
    n_blocks = np.ceil(float(n_points)/(block[0] * block[1]))
    g_cols = 2
    g_rows = np.int(np.ceil(n_blocks / g_cols))
    grid = (g_rows, g_cols, 1)

    start = cuda.Event()
    end = cuda.Event()

    start.record()  # start timing
    diameter3d(width, height, depth,
               n_points,
               norm_factor,
               max_iters,
               n_scan_angles,
               gpu_X,
               gpu_Y,
               gpu_Z,
               gpu_scan_angl_arr,
               gpu_azth_data_1d,
               gpu_lat_data_1d,
               gpu_radius_arr,
               block=block,
               grid=grid)
    end.record()  # end timing
    end.synchronize()

    dm_time = start.time_till(end)*1e-3
    print("Diameter estimation time: %fs" % (dm_time))

    radius_arr = gpu_radius_arr.get()

    if do_reshape:
        radius_arr = np.reshape(radius_arr, data.shape, order='C')

    out = {'diameter': radius_arr * 2., 'time': dm_time}
    return out


def estimate_diameter_single_run(name, output_dir, data, skel, lat_data, azth_data,
                                 n_scan_angles=32, make_output=True):
    """Computes 3D diameter using GPU and stores result in a npy file.

    Parameters
    ----------
    name : str
        Indicates the name of the output npy file.

    output_dir : str
        Indicates the path to the output folder where the data will be stored.

    data : 3D array
        Indicates the 3D binary data.

    skel : 3D array
        Indicates the skeleton of the binary data.

    lat_data : 3D array
        Indicates the 3D array containing latitude / elevation angle at every point of
        the skeleton in radians.

    azth_data : 3D array
        Indicates the 3D array containing azimuth angle at every point of the skeleton
        in radians.

    n_scan_angles : int
        Indicates the number of scanning angles on a range [0, 360] degrees.

    make_output : boolean
        Specifies if the estimated data should be stored.

    Returns
    -------
    output_props : dict
        The dictionary of properties specifying the sample name, the algorithm name,
        the number of processes, and the execution time.
    """
    if not cuda_available:
        print('The pycuda package is not found. The diameter estimation cannot be done.')
        return None

    output = dict()
    output_props = dict()

    output['diameter'] = np.zeros_like(data, dtype=np.float32)

    output_props['time'] = 0.
    output_props['sample_name'] = name
    output_props['type'] = 'diameter_gpu'
    output_props['n_processes'] = 1

    result = estimate_diameter_gpu(skel, data, lat_data, azth_data, n_scan_angles, do_reshape=False)

    output_props['time'] += result['time']
    output['diameter'][skel.nonzero()] = result['diameter']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('Total diameter execution time: {}'%format(output_props['time']))

    if make_output:
        opath = os.path.join(output_dir, '{}.npy').format(name)
        output_props['output_path'] = opath
        output['indices'] = skel > 0
        output['props'] = output_props
        np.save(opath, output)

    return output_props


def estimate_diameter_batches(name, output_dir, data, skel, lat_data, azth_data, border_gap,
                              n_scan_angles=32, out_arr_names=['diameter'], make_output=True,
                              slices_per_batch=100):
    """Computes 3D diameter using GPU in batches and stores result in a npy file.

    Parameters
    ----------
    name : str
        Indicates the name of the output npy file.

    output_dir : str
        Indicates the path to the output folder where the data will be stored.

    data : 3D array
        Indicates the 3D binary data.

    skel : 3D array
        Indicates the skeleton of the binary data.

    lat_data : 3D array
        Indicates the 3D array containing latitude / elevation angle at every point of
        the skeleton in radians.

    azth_data : 3D array
        Indicates the 3D array containing azimuth angle at every point of the skeleton
        in radians.

    border_gap : integer
        Indicates the number of overlapping slices along z-axis, usually it should be more or
        equal to a half of size of the 3D local window.

    n_scan_angles : int
        Indicates the number of scanning angles on a range [0, 360] degrees.

    out_arr_names : array of str
        Indicates the array of keys of the output dictionary.

    make_output : boolean
        Specifies if the estimated data should be stored.

    slices_per_batch : integer
        The number of slices along z-axis in a batch.

    Returns
    -------
    output_props : dict
        The dictionary of properties specifying the sample name, the algorithm name,
        the number of processes, and the execution time.
    """
    if not cuda_available:
        print('The pycuda package is not found. The diameter estimation cannot be done.')
        return None

    output = dict()
    output_props = dict()

    for arr_name in out_arr_names:
        output[arr_name] = np.zeros_like(data, dtype=np.float32)

    output_props['time'] = 0.
    output_props['sample_name'] = name
    output_props['type'] = 'diameter_gpu'
    output_props['n_processes'] = 1

    depth, height, width = data.shape
    batches_idxs = np.array_split(np.arange(depth), np.ceil(depth / float(slices_per_batch)))
    print(batches_idxs)

    for batch_idxs in batches_idxs:
        batch_len = len(batch_idxs)

        if batch_idxs[0] == 0:
            arr1, arr2 = None, np.arange(batch_idxs[-1]+1, batch_idxs[-1]+border_gap+1)
            gaped_batch_idxs = np.concatenate((batch_idxs, arr2))
        elif batch_idxs[-1] == (depth - 1):
            arr1, arr2 = np.arange(batch_idxs[0]-border_gap, batch_idxs[0]), None
            gaped_batch_idxs = np.concatenate((arr1, batch_idxs))
        else:
            arr1, arr2 = np.arange(batch_idxs[0]-border_gap, batch_idxs[0]), \
                            np.arange(batch_idxs[-1], batch_idxs[-1]+border_gap+1)
            gaped_batch_idxs = np.concatenate((arr1, batch_idxs, arr2))

        batched_data = data[gaped_batch_idxs]
        batched_skel = skel[gaped_batch_idxs]
        batched_lat_data = lat_data[gaped_batch_idxs]
        batched_azth_data = azth_data[gaped_batch_idxs]

        gaped_out_dict = estimate_diameter_gpu(batched_skel,
                                               batched_data,
                                               batched_lat_data,
                                               batched_azth_data,
                                               n_scan_angles,
                                               do_reshape=False)

        output_props['time'] += gaped_out_dict['time']

        for data_name in out_arr_names:
            gaped_arr_values = gaped_out_dict[data_name]
            gaped_arr = np.zeros_like(batched_data, dtype=np.float32)
            gaped_arr[batched_skel.nonzero()] = gaped_arr_values

            if batch_idxs[0] == 0:
                output[data_name][batch_idxs] = gaped_arr[:batch_len]
            elif batch_idxs[-1] == (depth - 1):
                output[data_name][batch_idxs] = gaped_arr[border_gap:]
            else:
                output[data_name][batch_idxs] = gaped_arr[border_gap:border_gap+batch_len]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('Total diameter execution time: {}'%(output_props['time']))

    if make_output:
        opath = os.path.join(output_dir, '{}.npy').format(name)
        output_props['output_path'] = opath
        output['indices'] = output['diameter'] > 0
        output['props'] = output_props
        np.save(opath, output)

    return output_props


def _calc_sphericity(volume, perimeter):
    """Computes sphericity from volume and perimeter of an object.
    """
    r = ((3.0 * volume) / (4.0 * np.pi)) ** (1.0/3.0)
    return (4.0 * np.pi * (r*r)) / perimeter


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

    print('Object counting - Labeling...')
    labeled_stack, num_labels = ndi.measurements.label(stack_binary_data)
    objects_stats = pd.DataFrame(columns=measurements_vals)

    print('Object counting - Stats gathering...')
    for slice_idx in np.arange(labeled_stack.shape[0]):
        for region in measure.regionprops(labeled_stack[slice_idx]):
            objects_stats = objects_stats.append({mname: region[mval]
                                        for mname, mval in _MEASUREMENTS.items()},
                                            ignore_index=True)

    print('Object counting - Stats grouping...')
    objects_stats = objects_stats.groupby('Label', as_index=False).sum()
    objects_stats['Sphericity'] = _calc_sphericity(objects_stats['Area'],
                                                   objects_stats['Perimeter'])

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
    out = {'Material {}'.format(m): \
                        1. - (float(data[data == m].size) / total_volume) \
                                                        for m in mats[mats > 0]}
    return out
