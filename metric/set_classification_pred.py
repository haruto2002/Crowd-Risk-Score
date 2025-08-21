import glob
import os
import json
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import yaml
from utils import get_cropped_crs, get_cropped_density, get_bev_points


def process_target_info(args):
    (
        i,
        target_info_path,
        crs_map_path_list,
        vec_data_path_list,
        grid_size,
        freq,
        path2homography_matrix,
    ) = args

    with open(target_info_path, "r") as f:
        target_info = json.load(f)

    frame_range = target_info["frame_range"]
    crop_area = target_info["crop_points"]
    target_crs_map_path_list = crs_map_path_list[
        frame_range[0] // freq : frame_range[1] // freq - 1
    ]
    target_vec_data_path_list = vec_data_path_list[
        frame_range[0] // freq : frame_range[1] // freq - 1
    ]
    bev_crop_data = get_bev_points(path2homography_matrix, crop_area)

    crs_frame_list = []
    density_frame_list = []

    for crs_map_path in target_crs_map_path_list:
        crs_map = np.loadtxt(crs_map_path)
        crs_map_cropped = get_cropped_crs(crs_map, bev_crop_data, grid_size)
        crs_frame_list.append(crs_map_cropped)

    for vec_data_path in target_vec_data_path_list:
        vec_data = np.loadtxt(vec_data_path)
        density = get_cropped_density(vec_data, bev_crop_data)
        density_frame_list.append(density)

    crs_mean = np.mean(crs_frame_list)
    density_mean = np.mean(density_frame_list)

    return i, {"crs": float(crs_mean), "density": float(density_mean)}


def set_classification_pred(path2dataset, pred_dir):
    path2cfg = os.path.join(pred_dir, "config.yaml")
    with open(path2cfg, "r") as f:
        cfg = yaml.safe_load(f)
    grid_size = cfg["grid_size"]
    path2homography_matrix = cfg["trajectory_dir"] + "/homography_matrix.txt"
    freq = cfg["freq"]
    crs_map_dir = os.path.join(pred_dir, "each_result", "crs_map")
    vec_data_dir = os.path.join(pred_dir, "each_result", "vec_data")
    crs_map_path_list = sorted(glob.glob(os.path.join(crs_map_dir, "*.txt")))
    vec_data_path_list = sorted(glob.glob(os.path.join(vec_data_dir, "*.txt")))

    print("Setting data for targets")

    target_info_list = sorted(glob.glob(os.path.join(path2dataset, "*.json")))
    print(f"Found {len(target_info_list)} target info files")

    pool_list = [
        (
            i,
            target_info_path,
            crs_map_path_list,
            vec_data_path_list,
            grid_size,
            freq,
            path2homography_matrix,
        )
        for i, target_info_path in enumerate(target_info_list)
    ]

    pool_size = os.cpu_count()
    print(f"Using {pool_size} processes for parallel processing")

    estimate_data = {}
    with Pool(processes=pool_size) as pool:
        with tqdm(total=len(pool_list), desc="Processing targets") as pbar:
            for i, result in pool.imap_unordered(process_target_info, pool_list):
                estimate_data[i] = result
                pbar.update()

    save_dir = f"{pred_dir}/pred_data"
    os.makedirs(save_dir, exist_ok=True)
    dataset_name = path2dataset.split("/")[-1]
    output_path = f"{save_dir}/{dataset_name}_pred_data.json"
    with open(output_path, "w") as f:
        json.dump(estimate_data, f)

    print(f"Results saved to {output_path}")
