import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def _sobel_operator_3d(img, size=3):
    if size == 3:
        sobel_filter_z = tf.constant(
            [
                [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[-1.0, -2.0, -1.0], [-2.0, -4.0, -2.0], [-1.0, -2.0, -1.0]],
            ]
        )
        sobel_filter_y = tf.transpose(sobel_filter_z, [1, 0, 2])
        sobel_filter_x = tf.transpose(sobel_filter_z, [2, 1, 0])

        sobel_filter_z = sobel_filter_z[..., tf.newaxis, tf.newaxis]
        sobel_filter_y = sobel_filter_y[..., tf.newaxis, tf.newaxis]
        sobel_filter_x = sobel_filter_x[..., tf.newaxis, tf.newaxis]
    else:
        raise ValueError("Only size of 3 is supported.")

    strides = [1, 1, 1, 1, 1]

    edges_z = tf.nn.conv3d(
        img,
        sobel_filter_z,
        strides,
        "SAME",
        data_format="NDHWC",
        dilations=None,
        name=None,
    )
    edges_y = tf.nn.conv3d(
        img,
        sobel_filter_y,
        strides,
        "SAME",
        data_format="NDHWC",
        dilations=None,
        name=None,
    )
    edges_x = tf.nn.conv3d(
        img,
        sobel_filter_x,
        strides,
        "SAME",
        data_format="NDHWC",
        dilations=None,
        name=None,
    )

    return edges_z, edges_y, edges_x


def _gaussian_kernel(size, mean, std):
    d = tfp.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))
    gauss_kernel = tf.einsum("i->i", vals)
    return gauss_kernel / tf.reduce_sum(gauss_kernel)


def _gaussian_filter(img, size, mean, std):
    kernel = _gaussian_kernel(size, mean, std)

    kernel_z = kernel[..., tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
    kernel_y = kernel[tf.newaxis, ..., tf.newaxis, tf.newaxis, tf.newaxis]
    kernel_x = kernel[tf.newaxis, tf.newaxis, ..., tf.newaxis, tf.newaxis]

    strides = [1, 1, 1, 1, 1]

    img_filt = tf.nn.conv3d(
        img, kernel_z, strides, "SAME", data_format="NDHWC", dilations=None, name=None
    )
    img_filt = tf.nn.conv3d(
        img_filt,
        kernel_y,
        strides,
        "SAME",
        data_format="NDHWC",
        dilations=None,
        name=None,
    )
    img_filt = tf.nn.conv3d(
        img_filt,
        kernel_x,
        strides,
        "SAME",
        data_format="NDHWC",
        dilations=None,
        name=None,
    )

    return img_filt


def _structure_tensor_3d(img, size=4, mean=0, std=2):
    G_z, G_y, G_x = _sobel_operator_3d(img, 3)

    Axx = _gaussian_filter(tf.math.multiply(G_x, G_x), size, mean, std)
    Axy = _gaussian_filter(tf.math.multiply(G_x, G_y), size, mean, std)
    Axz = _gaussian_filter(tf.math.multiply(G_x, G_z), size, mean, std)
    Ayy = _gaussian_filter(tf.math.multiply(G_y, G_y), size, mean, std)
    Ayz = _gaussian_filter(tf.math.multiply(G_y, G_z), size, mean, std)
    Azz = _gaussian_filter(tf.math.multiply(G_z, G_z), size, mean, std)

    return Axx, Axy, Axz, Ayy, Ayz, Azz


def local_3d_orientation_in_batch(img, size=4, mean=0, std=2):
    axes = np.arange(1, len(img.shape))
    Axx, Axy, Axz, Ayy, Ayz, Azz = _structure_tensor_3d(
        img, size=size, mean=mean, std=std
    )

    tensor_vals = tf.convert_to_tensor(
        [
            [
                tf.math.reduce_mean(Azz, axis=axes),
                tf.math.reduce_mean(Ayz, axis=axes),
                tf.math.reduce_mean(Axz, axis=axes),
            ],
            [
                tf.math.reduce_mean(Ayz, axis=axes),
                tf.math.reduce_mean(Ayy, axis=axes),
                tf.math.reduce_mean(Axy, axis=axes),
            ],
            [
                tf.math.reduce_mean(Axz, axis=axes),
                tf.math.reduce_mean(Axy, axis=axes),
                tf.math.reduce_mean(Axx, axis=axes),
            ],
        ]
    )
    tensor_vals = tensor_vals[::-1, ::-1, :]
    tensor_vals = tf.transpose(tensor_vals, [2, 0, 1])

    eigvals, eigvecs = tf.map_fn(
        tf.linalg.eig, tensor_vals, dtype=(tf.complex64, tf.complex64)
    )
    eps = tf.constant(1e-8)

    eigvals = tf.dtypes.cast(tf.math.real(eigvals), tf.float32)
    eigvecs = tf.transpose(tf.dtypes.cast(tf.math.real(eigvecs), tf.float32), [0, 2, 1])

    eigvals_min_indices = tf.dtypes.cast(tf.math.argmin(eigvals, axis=1), tf.int64)
    idxs = tf.stack(
        [tf.range(eigvals.shape[0], dtype=tf.int64), eigvals_min_indices], 1
    )
    mv = tf.gather_nd(eigvecs, idxs)

    bcst = tf.zeros(tf.shape(mv), dtype=mv.dtype)
    mv = tf.where(tf.less(tf.math.abs(mv), eps), bcst, mv)

    G = tf.math.sqrt(tf.math.square(mv[:, 2]) + tf.math.square(mv[:, 1]))
    lat = tf.math.asin(G)
    azth = tf.map_fn(
        lambda v: tf.math.atan(v[1] / v[2]) if v[2] else tf.constant(np.pi / 2.0),
        mv,
        dtype=tf.float32,
    )

    return lat, azth


def local_orientation(img, size=4, mean=0, std=2):
    Axx, Axy, Axz, Ayy, Ayz, Azz = _structure_tensor_3d(
        img, size=size, mean=mean, std=std
    )

    tensor_vals = tf.convert_to_tensor(
        [
            [
                tf.math.reduce_mean(Azz),
                tf.math.reduce_mean(Ayz),
                tf.math.reduce_mean(Axz),
            ],
            [
                tf.math.reduce_mean(Ayz),
                tf.math.reduce_mean(Ayy),
                tf.math.reduce_mean(Axy),
            ],
            [
                tf.math.reduce_mean(Axz),
                tf.math.reduce_mean(Axy),
                tf.math.reduce_mean(Axx),
            ],
        ]
    )
    tensor_vals = tensor_vals[::-1, ::-1]
    eigvals, eigvecs = tf.linalg.eig(tensor_vals)
    eps = tf.constant(1e-8)

    eigvals = tf.dtypes.cast(eigvals, tf.float32)
    eigvecs = tf.dtypes.cast(eigvecs, tf.float32)

    mv = eigvecs[:, np.argmin(eigvals)]  # z, y, x
    bcst = tf.zeros(tf.shape(mv), dtype=mv.dtype)
    mv = tf.where(tf.less(tf.math.abs(mv), eps), bcst, mv)

    G = tf.math.sqrt(tf.math.square(mv[2]) + tf.math.square(mv[1]))
    lat = tf.math.asin(G)
    azth = tf.math.atan(mv[1] / mv[2]) if mv[2] else tf.constant(np.pi / 2.0)

    return lat, azth


def _pad_to_size(data, ws):
    size = data.shape
    size_dx = [ws - v for v in size]
    pads = [(v // 2, v - v // 2) for v in size_dx]
    return np.pad(data, pads)


def _generate_3d_tiles_by_coords(skel, data, window_size):
    ws2 = window_size // 2
    Z, Y, X = np.nonzero(skel)

    for z, y, x in zip(Z, Y, X):
        data_local = data[z - ws2 : z + ws2, y - ws2 : y + ws2, x - ws2 : x + ws2]
        if any([v != window_size for v in data_local.shape]):
            data_local = _pad_to_size(data_local, window_size)
        yield np.array([z, y, x], dtype=np.uint64), data_local[:, :, :, tf.newaxis]


def estimate_3d_orientation(
    skel,
    data,
    window_size,
    n_channels=1,
    batch_size=100,
):
    azth_arr = np.zeros_like(data, dtype=np.float32)
    lat_arr = np.zeros_like(data, dtype=np.float32)

    ds = tf.data.Dataset.from_generator(
        _generate_3d_tiles_by_coords,
        args=[skel, data, window_size],
        output_types=(tf.uint64, tf.float32),
        output_shapes=((3,), (window_size, window_size, window_size, n_channels)),
    ).batch(batch_size)

    for coords, data_chunk in ds:
        z, y, x = coords.numpy().transpose()
        lat, azth = local_3d_orientation_in_batch(data_chunk)
        azth_arr[z, y, x] = azth.numpy()
        lat_arr[z, y, x] = lat.numpy()

    return lat_arr, azth_arr
