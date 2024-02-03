import numpy as np

import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda import gpuarray


def _raycasting_kernel():
    kernel_src = """
texture<float, cudaTextureType3D, cudaReadModeElementType> data_tex;

__device__ float4 approx_step(float4 cosin)
{
    float4 step;
    step.z = cosin.x;
    step.w = cosin.y;  // xy
    step.x = step.w * cosin.z; 
    step.y = step.w * cosin.w;
    return step;
}

__device__ float2 limit(int coord, int ws, int dim) {
    float2 limit;
    limit.x = coord - ws / 2;
    limit.y = limit.x + ws;
    
    limit.x = (limit.x < 0 ? 0 : limit.x) + 0.5;
    limit.y = (limit.y > dim ? dim : limit.y) - 0.5;
    
    return limit;
}

__device__ float trace (float x, float y, float z,
                        float4 step,
                        float2 x_lim, float2 y_lim, float2 z_lim)
{
    float cost = 0.0;
    float cx, cy, cz;
    cx = x + 0.5; cy = y + 0.5; cz = z + 0.5;
    while (cx >= x_lim.x && cx < x_lim.y
        && cy >= y_lim.x && cy < y_lim.y
        && cz >= z_lim.x && cz < z_lim.y)
    {
        cost += tex3D(data_tex, cx, cy, cz);
        cx += step.x;
        cy += step.y;
        cz += step.z;
    }
    
    cx = x - step.x + 0.5;
    cy = y - step.y + 0.5;
    cz = z - step.z + 0.5;
    while (cx >= x_lim.x && cx < x_lim.y
        && cy >= y_lim.x && cy < y_lim.y
        && cz >= z_lim.x && cz < z_lim.y)
    {
        cost += tex3D(data_tex, cx, cy, cz);
        cx -= step.x;
        cy -= step.y;
        cz -= step.z;
    }
    
    return cost;
}

__global__ void trace3d (unsigned int width,
                         unsigned int height,
                         unsigned int depth,
                         unsigned int ws,
                         int n_lat,
                         int n_azth,
                         int n_points,
                         const int *X,
                         const int *Y,
                         const int *Z,
                         const float *lat_arr,
                         const float *azth_arr,
                         float *best_cost_arr,
                         float *best_lat_arr,
                         float *best_azth_arr)
{
    unsigned long blockId, idx;
    blockId = blockIdx.x + blockIdx.y * gridDim.x;
    idx = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    
    if (idx > n_points) {
        return;
    }
    
    int x, y, z;
    x = X[idx]; y = Y[idx]; z = Z[idx];
    
    float2 x_lim, y_lim, z_lim;
    x_lim = limit(x, ws, width);
    y_lim = limit(y, ws, height);
    z_lim = limit(z, ws, depth);
    
    float lat, azth;
    float4 cosin, step;
    float cost = -1, best_cost = -1, best_lat = 0, best_azth = 0;
    for (int i = 0; i < n_lat; i++) {
        lat = lat_arr[i];
        cosin.x = cos(lat);
        cosin.y = sin(lat);
                
        for (int j = 0; j < n_azth; j++) {
            azth = azth_arr[j];
            //cosin.z = sin(azth);
            //cosin.w = cos(azth);
            cosin.z = cos(azth);
            cosin.w = sin(azth);
            
            step = approx_step(cosin);
            cost = trace(x, y, z, step, x_lim, y_lim, z_lim);
            if (cost >= best_cost) {
                best_cost = cost;
                best_azth = azth;
                best_lat = lat;
            }
        }
    }
    
    best_cost_arr[idx] = best_cost;
    best_azth_arr[idx] = best_azth;
    best_lat_arr[idx] = best_lat;
}
"""

    program = SourceModule(kernel_src)
    trace3d = program.get_function("trace3d")

    return program, trace3d


def _numpy3d_to_array(np_array, allow_surface_bind=True):
    # numpy3d_to_array
    # this function was
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


def estimate_3d_orientation_in_batches(
    skel,
    data,
    window_size,
    batch_size=100,
):
    output_lat, output_azth = np.zeros_like(data, dtype=np.float32)

    depth, _, _ = data.shape
    batches_idxs = np.array_split(np.arange(depth), np.ceil(depth / float(batch_size)))

    border_gap = window_size

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

        batched_skel = skel[gaped_batch_idxs]
        batched_data = data[gaped_batch_idxs]

        gaped_out_lat, gaped_out_azth = estimate_3d_orientation(
            batched_skel, batched_data, window_size, do_reshape=False
        )

        gaped_lat_arr = np.zeros_like(batched_data, dtype=np.float32)
        gaped_azth_arr = np.zeros_like(batched_data, dtype=np.float32)

        gaped_lat_arr[batched_skel.nonzero()] = gaped_out_lat
        gaped_azth_arr[batched_skel.nonzero()] = gaped_out_azth

        if batch_idxs[0] == 0:
            output_lat[batch_idxs] = gaped_lat_arr[:batch_len]
            output_azth[batch_idxs] = gaped_azth_arr[:batch_len]
        elif batch_idxs[-1] == (depth - 1):
            output_lat[batch_idxs] = gaped_lat_arr[border_gap:]
            output_azth[batch_idxs] = gaped_azth_arr[border_gap:]
        else:
            output_lat[batch_idxs] = gaped_lat_arr[border_gap : border_gap + batch_len]
            output_azth[batch_idxs] = gaped_azth_arr[
                border_gap : border_gap + batch_len
            ]

    return output_lat, output_azth


def estimate_3d_orientation(
    skel, data, window_size, n_lat=90, n_azth=180, do_reshape=True
):
    program, trace3d = _raycasting_kernel()

    Z, Y, X = np.int32(skel.nonzero())
    depth, height, width = np.uint32(skel.shape)
    data = np.array(data, dtype=np.float32)

    lat_arr = np.arange(0, np.pi / 2.0, np.pi / (2.0 * n_lat), dtype=np.float32)
    azth_arr = np.arange(-np.pi / 2.0, np.pi / 2.0, np.pi / n_azth, dtype=np.float32)

    best_cost_arr = np.zeros_like(X, dtype=np.float32)
    best_lat_arr = np.zeros_like(X, dtype=np.float32)
    best_azth_arr = np.zeros_like(X, dtype=np.float32)

    gpu_X = gpuarray.to_gpu(X)
    gpu_Y = gpuarray.to_gpu(Y)
    gpu_Z = gpuarray.to_gpu(Z)
    gpu_lat_arr = gpuarray.to_gpu(lat_arr)
    gpu_azth_arr = gpuarray.to_gpu(azth_arr)
    gpu_best_cost_arr = gpuarray.to_gpu(best_cost_arr)
    gpu_best_lat_arr = gpuarray.to_gpu(best_lat_arr)
    gpu_best_azth_arr = gpuarray.to_gpu(best_azth_arr)

    gpu_data_tex = program.get_texref("data_tex")
    gpu_data = _numpy3d_to_array(data)
    gpu_data_tex.set_array(gpu_data)

    n_points = np.uint32(len(Z))
    ws = np.uint32(window_size)
    n_lat = np.uint32(n_lat)
    n_azth = np.uint32(n_azth)

    block = (16, 16, 1)
    n_blocks = np.ceil(float(n_points) / (block[0] * block[1]))
    g_cols = 2
    g_rows = np.int(np.ceil(n_blocks / g_cols))
    grid = (g_rows, g_cols, 1)

    start = cuda.Event()
    end = cuda.Event()

    start.record()  # start timing
    trace3d(
        width,
        height,
        depth,
        ws,
        n_lat,
        n_azth,
        n_points,
        gpu_X,
        gpu_Y,
        gpu_Z,
        gpu_lat_arr,
        gpu_azth_arr,
        gpu_best_cost_arr,
        gpu_best_lat_arr,
        gpu_best_azth_arr,
        block=block,
        grid=grid,
    )

    end.record()  # end timing
    end.synchronize()

    # trace_time = start.time_till(end)*1e-3

    best_cost_arr = gpu_best_cost_arr.get()
    trace_lat_arr = gpu_best_lat_arr.get()
    trace_azth_arr = gpu_best_azth_arr.get()

    if do_reshape:
        best_cost_arr, trace_lat_arr, trace_azth_arr = (
            np.reshape(best_cost_arr, data.shape, order="C"),
            np.reshape(trace_lat_arr, data.shape, order="C"),
            np.reshape(trace_azth_arr, data.shape, order="C"),
        )

    return best_lat_arr, best_azth_arr
