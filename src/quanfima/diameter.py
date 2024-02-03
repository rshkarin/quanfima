import math

import numpy as np
from scipy.spatial import distance

from quanfima import cuda_available

if cuda_available:
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    from pycuda import gpuarray


def _3d_diameter_kernel():
    """Returns the CUDA kernel to estimate 3D diameter of structures in a 3D local window."""
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


def estimate_3d_fiber_diameter_gpu(
    skel, data, lat_data, azth_data, n_scan_angles, max_iters=150, do_reshape=True
):
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
        print(
            "The pycuda package is not found. The diameter estimation cannot be done."
        )
        return None

    program, diameter3d = _3d_diameter_kernel()

    Z, Y, X = np.int32(skel.nonzero())
    depth, height, width = np.uint32(skel.shape)

    scan_angl_arr = np.deg2rad(
        np.float32(np.linspace(0, 360, num=n_scan_angles, endpoint=False))
    )
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

    gpu_rad_tex = program.get_texref("tex_data")
    gpu_data = _numpy3d_to_array(data)
    gpu_rad_tex.set_array(gpu_data)

    n_scan_angles = np.uint32(n_scan_angles)
    n_points = np.uint32(len(Z))
    max_iters = np.uint32(max_iters)
    norm_factor = np.float32(1.0 / n_scan_angles)

    block = (16, 16, 1)
    n_blocks = np.ceil(float(n_points) / (block[0] * block[1]))
    g_cols = 2
    g_rows = np.int(np.ceil(n_blocks / g_cols))
    grid = (g_rows, g_cols, 1)

    start = cuda.Event()
    end = cuda.Event()

    start.record()  # start timing
    diameter3d(
        width,
        height,
        depth,
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
        grid=grid,
    )
    end.record()  # end timing
    end.synchronize()

    dm_time = start.time_till(end) * 1e-3
    print("Diameter estimation time: %fs" % (dm_time))

    radius_arr = gpu_radius_arr.get()

    if do_reshape:
        radius_arr = np.reshape(radius_arr, data.shape, order="C")

    return radius_arr * 2.0


def estimate_3d_fiber_diameter_gpu_batched(
    data,
    skel,
    lat_data,
    azth_data,
    border_gap,
    n_scan_angles=32,
    slices_per_batch=100,
):
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
        print(
            "The pycuda package is not found. The diameter estimation cannot be done."
        )
        return None

    output_diameter = np.zeros_like(data, dtype=np.float32)

    depth, _, _ = data.shape
    batches_idxs = np.array_split(
        np.arange(depth), np.ceil(depth / float(slices_per_batch))
    )

    for batch_idxs in batches_idxs:
        batch_len = len(batch_idxs)

        if batch_idxs[0] == 0:
            arr1, arr2 = None, np.arange(
                batch_idxs[-1] + 1, batch_idxs[-1] + border_gap + 1
            )
            gaped_batch_idxs = np.concatenate((batch_idxs, arr2))
        elif batch_idxs[-1] == (depth - 1):
            arr1, arr2 = np.arange(batch_idxs[0] - border_gap, batch_idxs[0]), None
            gaped_batch_idxs = np.concatenate((arr1, batch_idxs))
        else:
            arr1, arr2 = np.arange(
                batch_idxs[0] - border_gap, batch_idxs[0]
            ), np.arange(batch_idxs[-1], batch_idxs[-1] + border_gap + 1)
            gaped_batch_idxs = np.concatenate((arr1, batch_idxs, arr2))

        batched_data = data[gaped_batch_idxs]
        batched_skel = skel[gaped_batch_idxs]
        batched_lat_data = lat_data[gaped_batch_idxs]
        batched_azth_data = azth_data[gaped_batch_idxs]

        gaped_batch_diameter_data = estimate_3d_fiber_diameter_gpu(
            batched_skel,
            batched_data,
            batched_lat_data,
            batched_azth_data,
            n_scan_angles,
            do_reshape=False,
        )

        gaped_arr = np.zeros_like(batched_data, dtype=np.float32)
        gaped_arr[batched_skel.nonzero()] = gaped_batch_diameter_data

        if batch_idxs[0] == 0:
            output_diameter[batch_idxs] = gaped_arr[:batch_len]
        elif batch_idxs[-1] == (depth - 1):
            output_diameter[batch_idxs] = gaped_arr[border_gap:]
        else:
            output_diameter[batch_idxs] = gaped_arr[border_gap : border_gap + batch_len]

    return output_diameter


def _numpy3d_to_array(np_array, allow_surface_bind=True):
    """Converts 3D numpy array to 3D device array."""
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


def _cast_ray(theta, y0, x0, fiber_mask, ray_len=100):
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
        if np.abs(m) <= 1.0:
            y = int(y0 - i * m)
            x = int(x0 + i)
        else:
            y = int(y0 - i)
            x = int(x0 + i * 1.0 / m)

        if x < 0 or x >= fiber_mask.shape[1] or y < 0 or y >= fiber_mask.shape[0]:
            continue

        if fiber_mask[y, x] == 0:
            break

        y1, x1 = y, x

    for i in range(ray_len):
        if np.abs(m) <= 1.0:
            y = int(y0 - i * m)
            x = int(x0 + i)
        else:
            y = int(y0 - i)
            x = int(x0 + i * 1.0 / m)

        if x < 0 or x >= fiber_mask.shape[1] or y < 0 or y >= fiber_mask.shape[0]:
            continue

        if fiber_mask[y, x] == 0:
            break

        y2, x2 = y, x

    return distance.euclidean((y1, x1), (y2, x2))


def _scan_fiber_thickness(
    angle, patch, angular_step=1.0, tilt_range=range(-5, 6), ray_len=100
):
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

    y0, x0 = patch.shape[0] / 2, patch.shape[1] / 2
    pi2 = np.pi / 2.0
    rad_p_deg = np.deg2rad(angular_step)

    for tilt_amount in tilt_range:
        theta = angle + pi2 + tilt_amount * rad_p_deg
        thickness_dist.append(_cast_ray(theta, y0, x0, patch, ray_len=ray_len))

    average_distance = np.mean(thickness_dist)
    return average_distance


def estimate_2d_fiber_diameter(
    fiber_orientation, fiber_mask, paddding=25, window_radius=12
):
    padded_fiber_mask = np.pad(
        fiber_mask, pad_width=(paddding,), mode="constant", constant_values=0
    )
    padded_fiber_orientation = np.pad(
        fiber_orientation, pad_width=(paddding,), mode="constant", constant_values=0
    )

    output_padded_diameter_map = np.zeros_like(
        padded_fiber_orientation, dtype=np.float32
    )

    ycords, xcords = np.nonzero(padded_fiber_orientation)

    for _, (yy, xx) in enumerate(zip(ycords, xcords)):
        patch_mask = padded_fiber_mask[
            (yy - window_radius) : (yy + window_radius + 1),
            (xx - window_radius) : (xx + window_radius + 1),
        ]

        final_thickness = _scan_fiber_thickness(padded_fiber_orientation[yy, xx], patch_mask)
        output_padded_diameter_map[yy, xx] = final_thickness

    output_diameter_map = output_padded_diameter_map[
        paddding:-paddding, paddding:-paddding
    ]

    return output_diameter_map
