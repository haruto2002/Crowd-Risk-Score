import glob
import os
from multiprocessing import Pool

import cv2
import numpy as np
from tqdm import tqdm


class BackImgSingleGenerator:
    def __init__(
        self,
        img_dir: str,
        path2bev_matrix: str | None,
        path2bev_size: str | None,
        crop_area: list[int] | None,
    ):
        assert os.path.exists(img_dir), "img_dir does not exist"
        self.img_dir = img_dir

        assert (path2bev_matrix is None and path2bev_size is None) or (
            path2bev_matrix is not None and path2bev_size is not None
        ), "path2bev_matrix and path2bev_size must both be str or both be None"
        if path2bev_matrix is not None:
            assert os.path.exists(path2bev_matrix), "path2bev_matrix does not exist"
            assert os.path.exists(path2bev_size), "path2bev_size does not exist"
            self.matrix = np.loadtxt(path2bev_matrix)
            self.size = np.loadtxt(path2bev_size).astype(int)
        else:
            self.matrix = None
            self.size = None

        self.crop_area = crop_area

    def get_back_img(self, end_frame: int):
        path2img = sorted(glob.glob(f"{self.img_dir}/*.jpg"))[end_frame - 1]
        img = cv2.imread(path2img)
        if self.matrix is not None:
            img = cv2.warpPerspective(
                img,
                self.matrix,
                (self.size[0], self.size[1]),
                borderValue=(255, 255, 255),
            )

        if self.crop_area is not None:
            img = img[
                self.crop_area[1] : self.crop_area[3],
                self.crop_area[0] : self.crop_area[2],
                :,
            ]

        return img


class BackImgGenerator:
    def __init__(self, img_dir, path2bev_matrix, path2bev_size, crop_area, name_list):
        self.img_dir = img_dir
        self.path2bev_matrix = path2bev_matrix
        self.path2bev_size = path2bev_size
        self.crop_area = crop_area
        self.name_list = name_list

    def run(self):
        back_img_generator = BackImgSingleGenerator(
            self.img_dir, self.path2bev_matrix, self.path2bev_size, self.crop_area
        )
        pool_size = min(len(self.name_list), os.cpu_count())
        pool_list = []
        for name in self.name_list:
            last_frame = int(name.split("_")[-1])
            pool_list.append((back_img_generator, last_frame))
        with Pool(pool_size) as p:
            back_img_results = list(
                tqdm(
                    p.imap_unordered(self.parallel_get_back_img, pool_list),
                    total=len(pool_list),
                )
            )
        sorted_back_img_results = sorted(back_img_results, key=lambda x: x[0])
        back_img_list = [img for _, img in sorted_back_img_results]
        return back_img_list

    def parallel_get_back_img(self, inputs):
        (back_img_generator, end_frame) = inputs
        return end_frame, back_img_generator.get_back_img(end_frame)
