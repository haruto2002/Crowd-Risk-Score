import os
from dataclasses import dataclass
from multiprocessing import Pool

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class HeatmapConfig:
    grid_size: int
    max_score: float
    min_score: float
    scale: int
    save_dir: str
    display_vec: bool
    resize_method: str


class HeatmapGenerator:
    def __init__(
        self,
        heatmap_cfg: HeatmapConfig,
    ):
        self.grid_size = heatmap_cfg.grid_size
        self.max_score = heatmap_cfg.max_score
        self.min_score = heatmap_cfg.min_score
        self.scale = heatmap_cfg.scale
        self.save_dir = heatmap_cfg.save_dir
        self.display_vec = heatmap_cfg.display_vec
        self.resize_method = heatmap_cfg.resize_method

    def run(self, inputs):
        (i, map_data, vec_data, img, name) = inputs
        return i, self.display_single(map_data, vec_data, img, name)

    def display_single(self, map_data, vec_data, img, name):
        img = cv2.resize(img, None, fx=self.scale, fy=self.scale)
        img_height, img_width, _ = img.shape
        vec_data *= self.scale

        resized_map_data = self.resize_map_data(map_data, img_width, img_height)

        min_score = self.min_score
        max_score = self.max_score
        # NaN を最小値に置き換え
        resized_map_data[np.isnan(resized_map_data)] = min_score
        # min_score, max_score を用いた正規化
        resized_map_data = np.clip(resized_map_data, min_score, max_score)
        map_norm_map = (
            (resized_map_data - min_score) / (max_score - min_score) * 255
        ).astype(np.uint8)
        map_heatmap = cv2.applyColorMap(map_norm_map, cv2.COLORMAP_JET)

        # 矢印の描画
        if self.display_vec:
            arrow_len = 5
            tipLength = 0.3
            arrow_scale = 50
            # Heatmap is in front
            for x, y, vx, vy in vec_data:
                # ベクトルを圧縮・スケール
                vec = np.array([vx, vy], dtype=float)
                vec = self.compress_vec_data(vec) * arrow_scale

                # 開始点・終了点
                start = (int(x), int(y))
                end = (int(x + vec[0]), int(y + vec[1]))

                # 角度を計算（-π～+π）
                angle = np.arctan2(vec[1], vec[0])

                # Hue を 0–179 にマッピング
                hue = ((angle + np.pi) / (2 * np.pi) * 179).astype(int)
                # HSV 画像（1×1）を作って BGR に変換
                hsv_pixel = np.uint8([[[hue, 255, 255]]])  # H, S=255, V=255
                bgr_color = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0].tolist()

                # 矢印を描画
                cv2.arrowedLine(
                    img,
                    start,
                    end,
                    color=bgr_color,
                    thickness=arrow_len,
                    tipLength=tipLength,
                )
        img = img.astype(np.uint8)
        output = cv2.addWeighted(img, 0.5, map_heatmap, 0.5, 0)

        # 保存
        hm_vis_save_dir = self.save_dir + "/hm_vis_" + str(img_height)
        hm_raw_save_dir = self.save_dir + "/hm_raw_" + str(img_height)
        os.makedirs(hm_vis_save_dir, exist_ok=True)
        os.makedirs(hm_raw_save_dir, exist_ok=True)
        hm_vis_save_path = os.path.join(hm_vis_save_dir, f"{name}.png")
        hm_raw_save_path = os.path.join(hm_raw_save_dir, f"{name}.png")
        cv2.imwrite(hm_vis_save_path, output)
        cv2.imwrite(hm_raw_save_path, map_heatmap)

        return output

    def resize_map_data(self, map_data, img_width, img_height):
        if self.resize_method == "linear":
            resized_map_data = cv2.resize(
                map_data,
                (img_width, img_height),
                interpolation=cv2.INTER_LINEAR,
            )

        elif self.resize_method == "nearest":
            resized_map_data = np.zeros((img_height, img_width))
            for i in range(map_data.shape[0]):
                for j in range(map_data.shape[1]):
                    resized_map_data[
                        i * self.grid_size * self.scale : (i + 1)
                        * self.grid_size
                        * self.scale,
                        j * self.grid_size * self.scale : (j + 1)
                        * self.grid_size
                        * self.scale,
                    ] = map_data[i, j]
        else:
            raise ValueError(f"Invalid resize method: {self.resize_method}")

        return resized_map_data

    def compress_vec_data(self, vel):
        norm = np.linalg.norm(vel)
        over_ratio = norm / 0.35
        if norm > 0.35:
            vel = vel / norm * 0.35 * (over_ratio * 0.5)
        return vel


class HeatmapVisualizer:
    def __init__(
        self,
        heatmap_cfg: HeatmapConfig,
        map_data_list: list[np.ndarray],
        vec_data_list: list[np.ndarray],
        img_list: list[np.ndarray],
        name_list: list[str],
    ):
        self.heatmap_generator = HeatmapGenerator(heatmap_cfg)
        self.map_data_list = map_data_list
        self.vec_data_list = vec_data_list
        self.img_list = img_list
        self.name_list = name_list

    def display_heatmap_parallel(self):
        pool_list = []
        for i, (map_data, vec_data, img, name) in enumerate(
            zip(self.map_data_list, self.vec_data_list, self.img_list, self.name_list)
        ):
            inputs = (i, map_data, vec_data, img, name)
            pool_list.append(inputs)

        pool_size = min(len(pool_list), os.cpu_count())
        with Pool(pool_size) as p:
            results = list(
                tqdm(
                    p.imap_unordered(self.heatmap_generator.run, pool_list),
                    total=len(pool_list),
                )
            )

        sorted_results = sorted(results, key=lambda x: x[0])
        heatmap_img_list = [img for _, img in sorted_results]

        return heatmap_img_list


def get_global_min_max(map_data_list, dense=False):
    max_values = [hm.max() for hm in map_data_list]
    min_values = [hm.min() for hm in map_data_list]

    # 降順ソートして上位 100 個を取得
    top = sorted(max_values, reverse=True)[:50]
    bottom = sorted(min_values, reverse=False)[:50]

    # 上位 100 個の平均を計算
    max_mean = np.mean(top)
    min_mean = np.mean(bottom)

    return max_mean, min_mean
