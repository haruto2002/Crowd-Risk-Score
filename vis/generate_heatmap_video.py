import argparse
import glob
import os

import cv2
import numpy as np
import yaml
from bach_img_generator import BackImgGenerator
from heatmap_visualizer import HeatmapConfig, HeatmapVisualizer, get_global_min_max
from tqdm import tqdm


# SET DATA
def set_data(
    danger_source_dir, vec_source_dir, frame_range, freq, crop_area, grid_size
):
    map_crop_area = [
        crop_area[0] // grid_size,
        crop_area[1] // grid_size,
        crop_area[2] // grid_size,
        crop_area[3] // grid_size,
    ]

    crs_map_path_list = sorted(glob.glob(os.path.join(danger_source_dir, "*.txt")))
    vec_data_path_list = sorted(glob.glob(os.path.join(vec_source_dir, "*.txt")))

    target_crs_map_path_list = crs_map_path_list[
        frame_range[0] // freq : frame_range[1] // freq
    ]
    target_vec_data_path_list = vec_data_path_list[
        frame_range[0] // freq : frame_range[1] // freq
    ]

    crs_map_list = [
        crop_map(np.loadtxt(crs_map_path), map_crop_area)
        for crs_map_path in target_crs_map_path_list
    ]
    vec_data_list = [
        crop_vec_data(np.loadtxt(vec_data_path), crop_area)
        for vec_data_path in target_vec_data_path_list
    ]
    name_list = [
        os.path.basename(crs_map_path).split(".")[0]
        for crs_map_path in target_crs_map_path_list
    ]
    return crs_map_list, vec_data_list, name_list


def crop_map(map_data, crop_area):
    return map_data[crop_area[1] : crop_area[3], crop_area[0] : crop_area[2]]


def crop_vec_data(vec_data, crop_area):
    crop_vec_data = vec_data[
        (vec_data[:, 0] >= crop_area[0])
        & (vec_data[:, 0] <= crop_area[2])
        & (vec_data[:, 1] >= crop_area[1])
        & (vec_data[:, 1] <= crop_area[3])
    ]
    crop_vec_data[:, 0] -= crop_area[0]
    crop_vec_data[:, 1] -= crop_area[1]
    return crop_vec_data


# CREATE MOVIE
def create_movie_from_img_list(img_list, save_dir, name):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        os.path.join(save_dir, f"{name}.mp4"),
        fourcc,
        10.0,
        (img_list[0].shape[1], img_list[0].shape[0]),
    )
    for img in tqdm(img_list):
        out.write(img)
    out.release()


def create_movie_from_img_dir(img_dir, save_dir, name):
    img_list = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    img_list = [cv2.imread(img_path) for img_path in img_list]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        os.path.join(save_dir, f"{name}.mp4"),
        fourcc,
        10.0,
        (img_list[0].shape[1], img_list[0].shape[0]),
    )
    for img in tqdm(img_list):
        out.write(img)
    out.release()


# VISUALIZE HEATMAP
def visualize_heatmap(
    save_base_dir,
    img_dir,
    source_dir,
    frame_range=[801, 1500],
    crop_area=[180, 350, 460, 630],
    path2bev_matrix=None,
    path2bev_size=None,
    scale=5,
    min_score=0,
    max_score=None,
    resize_method="linear",
    display_vec=True,
):
    print("Setting data...")
    path2cfg = os.path.join(source_dir, "config.yaml")
    with open(path2cfg, "r") as f:
        cfg = yaml.safe_load(f)
    grid_size = cfg["grid_size"]
    freq = cfg["freq"]
    danger_source_dir = os.path.join(source_dir, "each_result", "crs_map")
    vec_source_dir = os.path.join(source_dir, "each_result", "vec_data")
    save_dir = f"{save_base_dir}/{frame_range[0]}_{frame_range[1]}"
    os.makedirs(save_dir, exist_ok=True)

    crs_map_list, vec_data_list, name_list = set_data(
        danger_source_dir, vec_source_dir, frame_range, freq, crop_area, grid_size
    )

    print("Generating background image...")
    back_img_generator = BackImgGenerator(
        img_dir, path2bev_matrix, path2bev_size, crop_area, name_list
    )
    back_img_list = back_img_generator.run()

    print("Displaying heatmap...")
    global_min_score, global_max_score = get_global_min_max(crs_map_list)
    if min_score is None:
        min_score = global_min_score
    if max_score is None:
        max_score = global_max_score
    heatmap_cfg = HeatmapConfig(
        grid_size=grid_size,
        max_score=max_score,
        min_score=min_score,
        scale=scale,
        save_dir=save_dir,
        display_vec=display_vec,
        resize_method=resize_method,
    )
    heatmap_visualizer = HeatmapVisualizer(
        heatmap_cfg, crs_map_list, vec_data_list, back_img_list, name_list
    )
    heatmap_img_list = heatmap_visualizer.display_heatmap_parallel()

    create_movie_from_img_list(heatmap_img_list, save_dir, "heatmap")


# RUN MAIN
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_base_dir", type=str, default="results_202601/GC_0001")
    parser.add_argument("--img_dir", type=str, default="trajectory_data/GC_0001/img")
    parser.add_argument("--source_dir", type=str, default="results_202601/GC_0001")
    parser.add_argument(
        "--path2bev_matrix",
        type=str,
        default="trajectory_data/GC_0001/homography_matrix.txt",
    )
    parser.add_argument(
        "--path2bev_size",
        type=str,
        default="trajectory_data/GC_0001/map_size.txt",
    )
    parser.add_argument(
        "--frame_range",
        type=int,
        nargs=2,
        default=None,
        help="frame range as two integers: start end",
    )
    parser.add_argument(
        "--crop_area",
        type=int,
        nargs=4,
        default=None,
        help="crop area as four integers: x1 y1 x2 y2",
    )
    parser.add_argument("--scale", type=int, default=5)
    return parser.parse_args()


def main():
    args = get_args()
    visualize_heatmap(
        args.save_base_dir,
        args.img_dir,
        args.source_dir,
        frame_range=args.frame_range,
        crop_area=args.crop_area,
        path2bev_matrix=args.path2bev_matrix,
        path2bev_size=args.path2bev_size,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()
