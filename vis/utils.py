import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def save_colorbar(save_dir, min_val, max_val, height=400, width=50):
    """Function to generate and save colorbar"""
    # Create colorbar
    gradient = np.linspace(min_val, max_val, height)[:, np.newaxis]
    gradient = np.tile(gradient, (1, width))

    # 正規化して0-255の範囲に変換
    gradient_norm = ((gradient - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    gradient_norm = np.tile(gradient_norm[:, :, np.newaxis], (1, 1, 3))

    # Apply colormap
    colorbar = cv2.applyColorMap(gradient_norm, cv2.COLORMAP_JET)

    # 保存
    colorbar_dir = os.path.join(save_dir, "colorbar")
    os.makedirs(colorbar_dir, exist_ok=True)
    cv2.imwrite(os.path.join(colorbar_dir, "colorbar.png"), colorbar)

    # Colorbar with text (using matplotlib)
    plt.figure(figsize=(2, 8))
    plt.imshow(gradient, cmap="jet")
    plt.colorbar()
    plt.axis("off")
    plt.savefig(
        os.path.join(colorbar_dir, "colorbar_with_text.png"),
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close()


def search_vec_data(vec_data_list):
    norm_data = [np.linalg.norm(data[:, 2:], axis=1) for data in vec_data_list]
    plt.hist(norm_data, bins=100)
    plt.savefig("BMVC/vec_data_main.jpg")
