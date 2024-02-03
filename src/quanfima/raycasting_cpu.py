import time

import numpy as np
from numba import jit, prange


@jit(nopython=True, fastmath=True)
def _interp_3d_slice_xyz(data, x, y, z):
    sz, sy, sx = data.shape
    # get z coord and validate it
    z_ = np.floor(z)
    if z_ >= sz:
        return 0

    # coordinates are logically separated from the data access
    # x_, y_ = round(x), round(y)
    x_, y_ = np.ceil(x - 0.5), np.ceil(y - 0.5)
    x_prev, x_next = x_ - 0.5, x_ + 0.5
    y_prev, y_next = y_ - 0.5, y_ + 0.5
    v, v1, v2 = 0, 0, 0

    ## interplate X direction
    w_prev = 1 - (x - x_prev)
    w_next = 1 - (x_next - x)

    slc = int(z_)
    if y_prev >= 0.5:
        row, col = int(y_prev - 0.5), int(x_prev - 0.5)
        v1 += w_prev * data[col, row, slc] if x_prev >= 0.5 else 0

        row, col = int(y_prev - 0.5), int(x_next - 0.5)
        v1 += w_next * data[col, row, slc] if x_next - 0.5 < sx else 0

    if y_next - 0.5 < sy:
        row, col = int(y_next - 0.5), int(x_prev - 0.5)
        v2 += w_prev * data[col, row, slc] if x_prev >= 0.5 else 0

        row, col = int(y_next - 0.5), int(x_next - 0.5)
        v2 += w_next * data[col, row, slc] if x_next - 0.5 < sx else 0

    ## interpolate Y direction
    w_prev = 1 - (y - y_prev)
    w_next = 1 - (y_next - y)
    v = v1 * w_prev + v2 * w_next
    return v


@jit(nopython=True, fastmath=True)
def _interp_3d(data, x, y, z):
    sz, _, _ = data.shape
    z_ = np.round(z)
    z_prev, z_next = z_ - 0.5, z_ + 0.5
    w_prev = 1 - (z - z_prev)
    w_next = 1 - (z_next - z)

    v1 = _interp_3d_slice_xyz(data, x, y, z_prev)
    v2 = _interp_3d_slice_xyz(data, x, y, z_next)

    v = v1 * w_prev + v2 * w_next

    return v


@jit(nopython=True, fastmath=True)
def _approx_step(cosin):
    return cosin[1] * cosin[2], cosin[1] * cosin[3], cosin[0], cosin[1]


@jit(nopython=True, fastmath=True)
def _limit(coord, ws, dim):
    lim = [0, 0]

    lim[0] = coord - ws // 2
    lim[1] = lim[0] + ws

    lim[0] = (0 if lim[0] < 0 else lim[0]) + 0.5
    lim[1] = (dim if lim[1] > dim else lim[1]) - 0.5

    return lim


@jit(nopython=True, fastmath=True, nogil=True)
def _trace(data, x, y, z, step, x_lim, y_lim, z_lim):
    #     grid = ((0.0, 511.0, 512), (0.0, 511.0, 512), (0.0, 511.0, 512))

    cost = 0.0

    cx = x + 0.5
    cy = y + 0.5
    cz = z + 0.5

    # while cx >= x_lim[0] and cx < x_lim[1] and cy >= y_lim[0] and cy < y_lim[1] and cz >= z_lim[0] and cz < z_lim[1]:
    for i in prange(15):
        cost += _interp_3d(
            data, cx, cy, cz
        )  # eval_linear(grid, data, np.array([[cz, cy, cx]]))[0] #_interp_3d(data, cx, cy, cz) # #
        cx += step[0]
        cy += step[1]
        cz += step[2]

    cx = x - step[0] + 0.5
    cy = y - step[1] + 0.5
    cz = z - step[2] + 0.5

    # while cx >= x_lim[0] and cx < x_lim[1] and cy >= y_lim[0] and cy < y_lim[1] and cz >= z_lim[0] and cz < z_lim[1]:
    for i in prange(15):
        cost += _interp_3d(
            data, cx, cy, cz
        )  # eval_linear(grid, data, np.array([[cz, cy, cx]]))[0] #_interp_3d(data, cx, cy, cz) # #;
        cx -= step[0]
        cy -= step[1]
        cz -= step[2]

    return cost


@jit(nopython=True, fastmath=True, nogil=True)
def _trace_new(data, x, y, z, step, n_steps=15):
    cost = 0.0

    cx1 = x + 0.5
    cy1 = y + 0.5
    cz1 = z + 0.5

    cx2 = x - step[0] + 0.5
    cy2 = y - step[1] + 0.5
    cz2 = z - step[2] + 0.5

    for i in prange(n_steps):
        cost += _interp_3d(data, cx1, cy1, cz1)
        cost += _interp_3d(data, cx2, cy2, cz2)

        cx1 += step[0]
        cy1 += step[1]
        cz1 += step[2]

        cx2 -= step[0]
        cy2 -= step[1]
        cz2 -= step[2]

    return cost


@jit(nopython=True, fastmath=True, parallel=True, nogil=True)
def estimate_3d_orientation_in_single_batch(
    skel, data, ws, lat_arr, azth_arr, lat_cos, lat_sin, azth_cos, azth_sin, Z, Y, X
):
    #     Z, Y, X = skel.nonzero()
    #     depth, height, width = np.uint32(data.shape)

    best_azth_arr = np.zeros_like(data)
    best_lat_arr = np.zeros_like(data)

    for idx in prange(len(X)):
        x = X[idx]
        y = Y[idx]
        z = Z[idx]

        cost, best_cost, best_lat, best_azth = -1, -1, 0, 0

        #         x_lim = _limit(x, ws, width)
        #         y_lim = _limit(y, ws, height)
        #         z_lim = _limit(z, ws, depth)

        for i in prange(len(lat_arr)):
            cosin_x = lat_cos[i]
            cosin_y = lat_sin[i]

            for j in prange(len(azth_arr)):
                cosin_z = azth_cos[j]
                cosin_w = azth_sin[j]

                #                 angel_step = np.array([cosin_x, cosin_y, cosin_z, cosin_w])
                #                 step = _approx_step(angel_step)
                step = (cosin_y * cosin_z, cosin_y * cosin_w, cosin_x, cosin_y)
                cost = _trace_new(skel, x, y, z, step, n_steps=32)

                if idx == 1000:
                    # print(f"{x},{y},{z} step={step} cost={cost} azth={azth_arr[j]} lat={lat_arr[i]}")
                    print(x, y, x, "cost =", cost, "azth =", np.rad2deg(azth_arr[j]), "lat =", np.rad2deg(lat_arr[i]))
                
                if cost >= best_cost:
                    best_azth = azth_arr[j]
                    best_lat = lat_arr[i]
                    best_cost = cost



        if idx == 1000:
            print(x, y, x, "best cost =", best_cost, "azth =", np.rad2deg(best_azth), "lat =", np.rad2deg(best_lat))

        best_azth_arr[x, y, z] = best_azth
        best_lat_arr[x, y, z] = best_lat

    return best_lat_arr, best_azth_arr


def estimate_3d_orientation(data, skel, n_lat=60, n_azth=90):
    lat_arr = np.arange(0, np.pi / 2.0, np.pi / (2.0 * n_lat))
    azth_arr = np.arange(-np.pi / 2.0, np.pi / 2.0, np.pi / n_azth)

    lat_cos, lat_sin = np.cos(lat_arr), np.sin(lat_arr)
    azth_cos, azth_sin = np.cos(azth_arr), np.sin(azth_arr)

    Z, Y, X = skel.nonzero()

    start = time.time()
    lat_arr, azth_arr = estimate_3d_orientation_in_single_batch(
        skel, data, 32, lat_arr, azth_arr, lat_cos, lat_sin, azth_cos, azth_sin, Z, Y, X
    )
    end = time.time()
    elapsed_time = end - start
    print("Ray Casting CPU time (fast math): %s" % elapsed_time)

    return lat_arr, azth_arr
