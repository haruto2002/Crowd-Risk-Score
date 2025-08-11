import cv2
import numpy as np
from matplotlib.path import Path


def get_bev_points(path2homography_matrix, crop_area):
    tl_x, tl_y, br_x, br_y = crop_area
    tl_point = np.array([tl_x, tl_y])
    tr_point = np.array([br_x, tl_y])
    br_point = np.array([br_x, br_y])
    bl_point = np.array([tl_x, br_y])
    crop_area_points = np.array([tl_point, tr_point, br_point, bl_point]).astype(np.float32)

    matrix = np.loadtxt(path2homography_matrix)
    bev_crop_area = cv2.perspectiveTransform(
        crop_area_points.reshape(-1, 1, 2), matrix
    ).reshape(-1, 2)
    bev_tl_point = bev_crop_area[0]
    bev_tr_point = bev_crop_area[1]
    bev_br_point = bev_crop_area[2]
    bev_bl_point = bev_crop_area[3]
    bev_crop_data = np.array([bev_tl_point, bev_tr_point, bev_br_point, bev_bl_point])
    
    return bev_crop_data


def get_cropped_crs(crs_map, bev_crop_data, grid_size):
    # bev_crop_data の４点を地図グリッド座標に変換し、整数化
    bev_pts = np.array(bev_crop_data)            # shape=(4,2)
    map_pts = (bev_pts / grid_size).astype(int)  # shape=(4,2), int に切り捨て

    # Path で多角形を定義
    polygon = Path(map_pts)

    # crs_map の全画素を対象に Point-in-Polygon 判定
    h, w = crs_map.shape
    # 横方向が X, 縦方向が Y としてメッシュを構築
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    coords = np.vstack((xx.ravel(), yy.ravel())).T  # shape=(h*w, 2)
    mask = polygon.contains_points(coords).reshape(h, w)

    # マスクで抽出
    cropped_vals = crs_map[mask]             # 1次元配列
    num_points = len(cropped_vals)

    # score = np.sum(cropped_vals)
    clip_score = np.sum(np.clip(cropped_vals, 0, None))
    return clip_score


def calc_quadrilateral_area(bev_crop_data):
    x1, y1 = bev_crop_data[0]
    x2, y2 = bev_crop_data[1]
    x3, y3 = bev_crop_data[2]
    x4, y4 = bev_crop_data[3]
    area = abs(
        x1*y2 + x2*y3 + x3*y4 + x4*y1
      - (y1*x2 + y2*x3 + y3*x4 + y4*x1)
    ) * 0.5
    return area


def get_cropped_density(vec_data, bev_crop_data, real_scale=13.5):
    # bev_crop_data をそのまま多角形頂点として定義
    polygon = Path(np.asarray(bev_crop_data))

    # 各ベクトルの (x,y) がポリゴン内にあるか
    # vec_data[:, :2] で最初の２列を取り出す
    pos_data = vec_data[:, :2]
    mask = polygon.contains_points(pos_data)

    # マスクで抽出
    cropped_pos_data = pos_data[mask]
    num_people = len(cropped_pos_data)

    area = calc_quadrilateral_area(bev_crop_data)

    real_area = area / (real_scale * real_scale)  # 1m=13.5pixel

    density = num_people / real_area  # 1m^2あたりの人数

    return density