import numpy as np
import os
import yaml
from tqdm import tqdm
from multiprocessing import Pool
import argparse
from utils.get_track_data import get_all_track, get_all_vec
from utils.clac_crowd_risk_score import (
    get_crs_data,
)


def run_parallel(inputs):
    (
        res_save_dir,
        trajectory_dir,
        start_frame,
        grid_size,
        R,
        span,
        crop_area
    ) = inputs
    crs_map, density_map, vec_list = main_single(
        res_save_dir,
        trajectory_dir,
        start_frame,
        grid_size,
        R,
        span,
        crop_area
    )
    return crs_map, density_map, vec_list


def main_single(
    save_dir,
    trajectory_dir,
    start_frame,
    grid_size,
    R,
    span,
    crop_area
):
    end_frame = start_frame + span
    all_track = get_all_track(trajectory_dir, start_frame, end_frame)
    map_size = np.loadtxt(f"{trajectory_dir}/map_size.txt").astype(int)
    vec_list = get_all_vec(all_track, end_frame)
    vec_data = np.array(vec_list)

    crs_map, density_map = get_crs_data(
        map_size,
        vec_data,
        grid_size,
        R,
    )

    if crop_area is not None:
        crop_crs_map = crs_map[
            crop_area[1] // grid_size:crop_area[3] // grid_size,
            crop_area[0] // grid_size:crop_area[2] // grid_size,
        ]
        crop_density_map = density_map[
            crop_area[1] // grid_size:crop_area[3] // grid_size,
            crop_area[0] // grid_size:crop_area[2] // grid_size,
        ]
        crop_vec_data = vec_data[
            (vec_data[:, 0, 0] > crop_area[0])
            & (vec_data[:, 0, 0] < crop_area[2])
            & (vec_data[:, 0, 1] > crop_area[1])
            & (vec_data[:, 0, 1] < crop_area[3])
        ]
        crop_vec_data[:, 0, 0] = crop_vec_data[:, 0, 0] - crop_area[0]
        crop_vec_data[:, 0, 1] = crop_vec_data[:, 0, 1] - crop_area[1]

        crs_save_dir = save_dir + "/crs_map"
        os.makedirs(crs_save_dir, exist_ok=True)
        np.savetxt(
            crs_save_dir + f"/{start_frame:04d}_{end_frame:04d}.txt",
            crop_crs_map
        )
        vec_save_dir = save_dir + "/vec_data"
        os.makedirs(vec_save_dir, exist_ok=True)
        save_vec_data = crop_vec_data.reshape(-1, 4)
        np.savetxt(
            vec_save_dir + f"/{start_frame:04d}_{end_frame:04d}.txt",
            save_vec_data
        )
        return crop_crs_map, crop_density_map, crop_vec_data
    else:
        crs_save_dir = save_dir + "/crs_map"
        os.makedirs(crs_save_dir, exist_ok=True)
        np.savetxt(
            crs_save_dir + f"/{start_frame:04d}_{end_frame:04d}.txt",
                crs_map
            )
        vec_save_dir = save_dir + "/vec_data"
        os.makedirs(vec_save_dir, exist_ok=True)
        save_vec_data = vec_data.reshape(-1, 4)
        np.savetxt(
            vec_save_dir + f"/{start_frame:04d}_{end_frame:04d}.txt",
            save_vec_data
        )
        return crs_map, density_map, vec_data


def main(
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
    if "debug" in dir_name:
        exist_ok = True
    else:
        exist_ok = False

    start_frame, end_frame = frame_range
    save_dir = f"{results_base_dir_name}/{dir_name}"
    print("SAVE_DIR >> ", save_dir)
    res_save_dir = f"{save_dir}/each_result"
    os.makedirs(res_save_dir, exist_ok=exist_ok)

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

    pool_list = []
    save_name_list = []
    for frame in range(start_frame, end_frame, freq):
        input = (
            res_save_dir,
            trajectory_dir,
            frame,
            grid_size,
            R,
            vec_span,
            crop_area
        )
        pool_list.append(input)

        save_name = f"{frame:04d}_{frame+vec_span:04d}"
        save_name_list.append(save_name)

    # run parallel
    print("calculating")
    pool_size = min(os.cpu_count(), len(pool_list))
    with Pool(pool_size) as p:
        list(tqdm(p.imap_unordered(run_parallel, pool_list), total=len(pool_list)))


def save_config(
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
):
    config = {
        "results_base_dir_name": results_base_dir_name,
        "dir_name": dir_name,
        "trajectory_dir": trajectory_dir,
        "crop_area": crop_area,
        "frame_range": list(frame_range),
        "freq": freq,
        "R": R,
        "grid_size": grid_size,
        "vec_span": vec_span,
    }

    file_name = save_dir + "/config.yaml"
    with open(file_name, "w") as file:
        yaml.dump(config, file, default_flow_style=False)

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"Config saved to {file_name}\n")


def load_config(path2yaml):
    with open(path2yaml, "r") as file:
        config = yaml.safe_load(file)
    print(f"Config loaded from {path2yaml}")
    return config


def run_exp():
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
        main(
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
        func_para = [args.decay, args.max_x, args.clip_value]
        frame_range = (args.frame_start, args.frame_end)

        main(
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
        "--use_yaml", action="store_true", help="yamlファイルを使用するかどうか"
    )
    parser.add_argument(
        "--yaml_path",
        type=str,
        default="src/config/config.yaml",
        help="yamlファイルのパス",
    )
    parser.add_argument(
        "--results_base_dir_name",
        type=str,
        default="results",
        help="結果保存ディレクトリ名"
    )
    parser.add_argument(
        "--dir_name", type=str, default="0811_debug", help="出力ディレクトリ名"
    )
    parser.add_argument(
        "--trajectory_dir",
        type=str,
        default="trajectory_data/WorldPorter_202408_0001",
        help="軌道データのディレクトリ",
    )
    parser.add_argument(
        "--crop_area",
        type=str,
        default=None,
        help="クロップエリア"
    )
    parser.add_argument("--grid_size", type=int, default=5, help="グリッドサイズ")
    parser.add_argument("--vec_span", type=int, default=10, help="ベクトル計算のスパン")
    parser.add_argument("--freq", type=int, default=10, help="危険度計算のフレーム間隔")
    parser.add_argument("--R", type=float, default=13.5, help="Rパラメータ")
    parser.add_argument("--frame_start", type=int, default=1, help="開始フレーム")
    parser.add_argument("--frame_end", type=int, default=8990, help="終了フレーム")

    return parser


if __name__ == "__main__":
    run_exp()
