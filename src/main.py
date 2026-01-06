import argparse
import os
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from engine.map_data_generator import MapDataGenerator
from engine.vec_data_generator import VectorGenerator
from utils.conf_utils import load_config, save_config


class CrowdRiskScore:
    def __init__(
        self,
        vector_generator: VectorGenerator,
        map_data_generator: MapDataGenerator,
        save_dir: str,
        grid_size: int,
        crop_area: list[int] | None,
    ):
        self.vector_generator = vector_generator
        self.map_data_generator = map_data_generator
        self.save_dir = save_dir
        self.crop_area = crop_area
        self.grid_size = grid_size

    def run(
        self,
        start_frame,
        end_frame,
    ):
        vec_data = self.vector_generator.generate_vec_data(start_frame, end_frame)
        crs_map, density_map = self.map_data_generator.generate_map_data(vec_data)
        self.save_map_data(crs_map, density_map, vec_data, start_frame, end_frame)

    def save_map_data(self, crs_map, density_map, vec_data, start_frame, end_frame):
        if self.crop_area is not None:
            crs_map, density_map, vec_data = self.crop_data(
                crs_map, density_map, vec_data
            )

        crs_save_dir = self.save_dir + "/crs_map"
        os.makedirs(crs_save_dir, exist_ok=True)
        np.savetxt(
            crs_save_dir + f"/{start_frame:04d}_{end_frame:04d}.txt",
            crs_map,
        )
        density_save_dir = self.save_dir + "/density_map"
        os.makedirs(density_save_dir, exist_ok=True)
        np.savetxt(
            density_save_dir + f"/{start_frame:04d}_{end_frame:04d}.txt",
            density_map,
        )
        vec_save_dir = self.save_dir + "/vec_data"
        os.makedirs(vec_save_dir, exist_ok=True)
        save_vec_data = vec_data.reshape(-1, 4)
        np.savetxt(
            vec_save_dir + f"/{start_frame:04d}_{end_frame:04d}.txt", save_vec_data
        )

    def crop_data(self, crs_map, density_map, vec_data):
        cropped_crs_map = crs_map[
            self.crop_area[1] // self.grid_size : self.crop_area[3] // self.grid_size,
            self.crop_area[0] // self.grid_size : self.crop_area[2] // self.grid_size,
        ]
        cropped_density_map = density_map[
            self.crop_area[1] // self.grid_size : self.crop_area[3] // self.grid_size,
            self.crop_area[0] // self.grid_size : self.crop_area[2] // self.grid_size,
        ]
        cropped_vec_data = vec_data[
            (vec_data[:, 0, 0] > self.crop_area[0])
            & (vec_data[:, 0, 0] < self.crop_area[2])
            & (vec_data[:, 0, 1] > self.crop_area[1])
            & (vec_data[:, 0, 1] < self.crop_area[3])
        ]
        cropped_vec_data[:, 0, 0] = cropped_vec_data[:, 0, 0] - self.crop_area[0]
        cropped_vec_data[:, 0, 1] = cropped_vec_data[:, 0, 1] - self.crop_area[1]
        return cropped_crs_map, cropped_density_map, cropped_vec_data


def run_parallel(inputs):
    (executor, s_frame, e_frame) = inputs
    executor.run(s_frame, e_frame)


def run_experiment(
    results_base_dir_name,
    dir_name,
    trajectory_dir,
    crop_area,
    grid_size,
    vec_span,
    freq,
    R,
    frame_range,
):
    start_frame, end_frame = frame_range
    save_dir = f"{results_base_dir_name}/{dir_name}"
    print("SAVE_DIR >> ", save_dir)
    res_save_dir = f"{save_dir}/each_result"
    os.makedirs(res_save_dir, exist_ok=False)

    save_config(
        save_dir,
        results_base_dir_name,
        dir_name,
        trajectory_dir,
        crop_area,
        frame_range,
        freq,
        R,
        grid_size,
        vec_span,
    )

    map_size = np.loadtxt(f"{trajectory_dir}/map_size.txt").astype(int)

    vector_generator = VectorGenerator(trajectory_dir)
    map_data_generator = MapDataGenerator(map_size, grid_size, R)

    executor = CrowdRiskScore(
        vector_generator, map_data_generator, res_save_dir, grid_size, crop_area
    )

    pool_list = []
    for frame in range(start_frame, end_frame, freq):
        s_frame = frame
        e_frame = frame + vec_span
        inputs = (executor, s_frame, e_frame)
        pool_list.append(inputs)

    print("calculating")
    pool_size = min(os.cpu_count(), len(pool_list))
    with Pool(pool_size) as p:
        list(tqdm(p.imap_unordered(run_parallel, pool_list), total=len(pool_list)))


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.use_yaml:
        config = load_config(args.yaml_path)
        results_base_dir_name = config["results_base_dir_name"]
        dir_name = config["dir_name"]
        trajectory_dir = config["trajectory_dir"]
        crop_area = config["crop_area"]
        grid_size = config["grid_size"]
        vec_span = config["vec_span"]
        freq = config["freq"]
        R = config["R"]
        frame_range = config["frame_range"]
        run_experiment(
            results_base_dir_name=results_base_dir_name,
            dir_name=dir_name,
            trajectory_dir=trajectory_dir,
            crop_area=crop_area,
            grid_size=grid_size,
            vec_span=vec_span,
            freq=freq,
            R=R,
            frame_range=frame_range,
        )
    else:
        frame_range = (args.frame_start, args.frame_end)
        run_experiment(
            results_base_dir_name=args.results_base_dir_name,
            dir_name=args.dir_name,
            trajectory_dir=args.trajectory_dir,
            crop_area=args.crop_area,
            grid_size=args.grid_size,
            vec_span=args.vec_span,
            freq=args.freq,
            R=args.R,
            frame_range=frame_range,
        )


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--use_yaml", action="store_true", help="Whether to use a yaml config file"
    )
    parser.add_argument(
        "--yaml_path",
        type=str,
        default="src/config/config.yaml",
        help="Path to the yaml config file",
    )
    parser.add_argument(
        "--results_base_dir_name",
        type=str,
        default="results",
        help="Base directory name for saving results",
    )
    parser.add_argument(
        "--dir_name", type=str, default="demo", help="Output directory name"
    )
    parser.add_argument(
        "--trajectory_dir",
        type=str,
        default="trajectory_data/WP_0001",
        help="Directory of trajectory data",
    )
    parser.add_argument("--crop_area", type=str, default=None, help="Crop area")
    parser.add_argument("--grid_size", type=int, default=5, help="Grid size")
    parser.add_argument(
        "--vec_span", type=int, default=10, help="Vector calculation span"
    )
    parser.add_argument(
        "--freq", type=int, default=10, help="Frame interval for risk calculation"
    )
    parser.add_argument("--R", type=float, default=13.5, help="Measurement parameter")
    parser.add_argument("--frame_start", type=int, default=1, help="Start frame")
    parser.add_argument("--frame_end", type=int, default=8990, help="End frame")

    return parser


if __name__ == "__main__":
    main()
