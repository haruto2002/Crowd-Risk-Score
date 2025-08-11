import numpy as np
import cv2
import glob
import os

def get_all_track(trajectory_dir, start_frame, end_frame, bev_transform=True):
    assert os.path.exists(trajectory_dir), f"The track directory {trajectory_dir} does not exist."
    txt_files = sorted(
        glob.glob(f"{trajectory_dir}/track_frame_data/*.txt")
    )
    track_list = [
        np.loadtxt(path2txt,delimiter=",") for path2txt in txt_files[start_frame - 1 : end_frame]
    ]

    if start_frame > 1:
        pre_num = 90
        pre_start_frame = max(1, start_frame - pre_num)
        pre_track_list = [
            np.loadtxt(path2txt,delimiter=",")
            for path2txt in txt_files[pre_start_frame - 1 : start_frame - 1]
        ]
        track_list = pre_track_list + track_list
        start_frame = pre_start_frame

    all_track = []
    for i, track in enumerate(track_list):
        frame = start_frame + i
        track = np.concatenate([np.full((len(track), 1), frame), track], axis=1)
        all_track += list(track)
    all_track = np.array(all_track)

    if bev_transform:
        path2matrix = f"{trajectory_dir}/homography_matrix.txt"
        all_track = bev_trans(path2matrix, all_track)

    return all_track

def bev_trans(path2matrix, all_track):
    assert os.path.exists(path2matrix), f"The bev directory {path2matrix} does not exist."
    matrix = np.loadtxt(path2matrix)
    all_track[:, 2:4] = cv2.perspectiveTransform(
        all_track[:, 2:4].reshape(-1, 1, 2), matrix
    ).reshape(-1, 2)
    return all_track


def get_all_vec(all_track, end_frame):
    id_list = np.unique(all_track[all_track[:, 0] == end_frame][:, 1])
    pool_list = [(id, all_track) for id in id_list]

    # pool_size = os.cpu_count()
    # with Pool(pool_size) as p:
    #     vec_list = p.map(get_vec, pool_list)

    vec_list = []
    for input in pool_list:
        vec_list.append(get_vec(input))

    if len(vec_list) == 0:
        print(all_track.shape)
        print("no vec! Check the track data or coding")
        print(end_frame)
        print(pool_list)
        # raise ValueError("no vec! Check the track data or coding")
    return vec_list


def get_vec(pool):
    id, all_track = pool
    track = all_track[all_track[:, 1] == id]
    start_point = track[0, 2:]
    end_point = track[-1, 2:]
    vec = (end_point - start_point) / len(track)
    pos = end_point
    return (pos, vec)
