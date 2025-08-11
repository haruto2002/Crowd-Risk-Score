import numpy as np
from scipy.spatial.distance import cdist


def clac_div_origin(around_data, center_pos, distance_decay):
    vx_list = []
    vy_list = []
    for pos, vec in around_data:
        vx,vy=vec[0],vec[1]
        if pos[0] < center_pos[0]:
            vx *= -1
        if pos[1] < center_pos[1]:
            vy *= -1
        vx_list.append(vx)
        vy_list.append(vy)
    vx_array = np.array(vx_list)
    vy_array = np.array(vy_list)
    decay_vx_array = vx_array * distance_decay
    decay_vy_array = vy_array * distance_decay
    div = np.sum(decay_vx_array) + np.sum(decay_vy_array)
    return div

def calc_crs(around_data, center_pos, R):
    local_density, distance_decay = get_gaussian_kernel_density(center_pos, around_data[:,0],R=R)
    div = clac_div_origin(around_data, center_pos, distance_decay)
    return -div*local_density,local_density
    

def get_gaussian_kernel_density(eval_point,positions, R):
    if len(positions)==0:
        return 0,None
    if positions.ndim == 1:
        positions = np.expand_dims(positions, axis=0)

    distances = cdist(np.expand_dims(eval_point, axis=0), positions)  # shape: (1, N)


    kernel_vals = np.exp(-0.5 * (distances / R)**2) / (2 * np.pi * R**2)  # (1, N)

    density = np.sum(kernel_vals[0])

    return density, kernel_vals[0]


def get_crs_data(size, vec_data, grid_size, R):
    x_size, y_size = size
    assert x_size % grid_size == 0 and y_size % grid_size == 0
    x_bins = np.arange(0, x_size + 1, grid_size)
    y_bins = np.arange(0, y_size + 1, grid_size)
    crs_map = np.zeros((len(y_bins) - 1, len(x_bins) - 1))
    density_map = np.zeros((len(y_bins) - 1, len(x_bins) - 1))
    around_range = R*3
    for i in range(len(y_bins) - 1):
        for j in range(len(x_bins) - 1):
            center_pos = np.array(
                [
                    grid_size * (j + 1 / 2),
                    grid_size * (i + 1 / 2),
                ]
            )
            around_data = vec_data[
                (vec_data[:, 0, 0] < center_pos[0] + around_range)
                & (vec_data[:, 0, 0] > center_pos[0] - around_range)
                & (vec_data[:, 0, 1] < center_pos[1] + around_range)
                & (vec_data[:, 0, 1] > center_pos[1] - around_range)
            ]

            if len(around_data) == 0:
                crs_map[i, j] = 0
                density_map[i, j] = 0
                continue

            crs, local_density = calc_crs(around_data, center_pos, R)
            crs_map[i, j] = crs
            density_map[i, j] = local_density

    return crs_map, density_map