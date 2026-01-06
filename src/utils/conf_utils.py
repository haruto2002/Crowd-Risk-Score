import yaml


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
