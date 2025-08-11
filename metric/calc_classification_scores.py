import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import glob
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)


def load_gt_list(path2dataset):
    target_info_list = sorted(glob.glob(os.path.join(path2dataset, "*.json")))
    gt_list_A = []
    gt_list_B = []
    for target_info_path in target_info_list:
        with open(target_info_path, "r") as f:
            target_info = json.load(f)
        gt_list_A.append(target_info["GT_A"])
        gt_list_B.append(target_info["GT_B"])
    return gt_list_A, gt_list_B


def main(save_dir, path2dataset, path2pred, eval_column):
    gt_list_A, gt_list_B = load_gt_list(path2dataset)
    stats_dict_0 = {"total": len(gt_list_A)}
    safe_indices_A = [str(i) for i, val in enumerate(gt_list_A) if val == 0]
    danger_indices_A = [str(i) for i, val in enumerate(gt_list_A) if val == 1]
    safe_indices_B = [str(i) for i, val in enumerate(gt_list_B) if val == 0]
    danger_indices_B = [str(i) for i, val in enumerate(gt_list_B) if val == 1]
    stats_dict_1 = {
        "safe_A": len(safe_indices_A),
        "danger_A": len(danger_indices_A),
        "safe_B": len(safe_indices_B),
        "danger_B": len(danger_indices_B),
    }
    same_safe_indices = list(set(safe_indices_A) & set(safe_indices_B))
    same_danger_indices = list(set(danger_indices_A) & set(danger_indices_B))
    A_safe_B_danger_indices = list(set(safe_indices_A) & set(danger_indices_B))
    A_danger_B_safe_indices = list(set(danger_indices_A) & set(safe_indices_B))
    stats_dict_2 = {
        "same_safe": len(same_safe_indices),
        "same_danger": len(same_danger_indices),
        "A_safe_B_danger": len(A_safe_B_danger_indices),
        "A_danger_B_safe": len(A_danger_B_safe_indices),
    }

    print("---------------------------------")
    print_aligned_stats(stats_dict_0)
    print("---------------------------------")
    print_aligned_stats(stats_dict_1)
    print("---------------------------------")
    print(
        "same_rate",
        f"{(len(same_safe_indices) + len(same_danger_indices) )/ len(gt_list_A)*100:.2f}%",
        "/",
        "diff_rate",
        f"{(len(A_safe_B_danger_indices) + len(A_danger_B_safe_indices) )/ len(gt_list_A)*100:.2f}%",
    )
    print_aligned_stats(stats_dict_2)
    print("---------------------------------")

    # print(diff_indices)

    assert len(A_safe_B_danger_indices) + len(A_danger_B_safe_indices) + len(
        same_safe_indices
    ) + len(same_danger_indices) == len(gt_list_A)

    with open(path2pred, "r") as f:
        data = json.load(f)

    danger_score_data_list_safe = [data[i][eval_column] for i in same_safe_indices]
    danger_score_data_list_danger = [data[i][eval_column] for i in same_danger_indices]
    labels = np.concatenate(
        [
            np.zeros(len(danger_score_data_list_safe)),
            np.ones(len(danger_score_data_list_danger)),
        ]
    )
    danger_score_auc, danger_score_ap = calculate_metrics(
        danger_score_data_list_safe + danger_score_data_list_danger,
        labels,
        eval_column,
        save_dir,
    )
    display_hist(
        [danger_score_data_list_safe, danger_score_data_list_danger],
        ["safe", "danger"],
        eval_column,
        save_dir,
    )

    save_stats(
        danger_indices_A,
        safe_indices_A,
        danger_indices_B,
        safe_indices_B,
        same_danger_indices,
        same_safe_indices,
        danger_score_auc,
        danger_score_ap,
        save_dir,
        eval_column,
    )


def save_stats(
    danger_indices_A,
    safe_indices_A,
    danger_indices_B,
    safe_indices_B,
    same_danger_indices,
    same_safe_indices,
    danger_score_auc,
    danger_score_ap,
    save_dir,
    eval_column,
):
    danger_num_A = len(danger_indices_A)
    safe_num_A = len(safe_indices_A)
    danger_num_B = len(danger_indices_B)
    safe_num_B = len(safe_indices_B)
    same_danger_num = len(same_danger_indices)
    same_safe_num = len(same_safe_indices)
    with open(f"{save_dir}/{eval_column}_stats.txt", "w") as f:
        f.write(f"<A>\n")
        f.write(f"Danger: {danger_num_A}\n")
        f.write(f"Safe: {safe_num_A}\n")
        f.write(f"<B>\n")
        f.write(f"Danger: {danger_num_B}\n")
        f.write(f"Safe: {safe_num_B}\n")
        f.write(f"<Same>\n")
        f.write(f"Danger: {same_danger_num}\n")
        f.write(f"Safe: {same_safe_num}\n\n")
        f.write(f"<Metric scores on same>\n")
        f.write(f"AUC: {danger_score_auc}\n")
        f.write(f"AP: {danger_score_ap}\n")


def print_aligned_stats(stats_dict):
    # キーと値の最大長を取得
    max_key_length = max(len(str(k)) for k in stats_dict.keys())
    max_value_length = max(len(f"{v}") for v in stats_dict.values())

    # フォーマット文字列を作成
    format_str = f"{{:<{max(max_key_length, 20)}}}: {{:>{max_value_length}}}"

    for key, value in stats_dict.items():
        print(format_str.format(key, value))


def display_hist(score_data_list, label_list, eval_column, save_dir, save_data=False):
    # save data
    if save_data:
        save_data_dir = f"{save_dir}/data"
        os.makedirs(save_data_dir, exist_ok=True)
        for i, score_data in enumerate(score_data_list):
            np.save(
                f"{save_data_dir}/{eval_column}_{label_list[i]}_score_data.npy",
                np.array(score_data),
            )
    # display histogram
    fig, ax = plt.subplots(figsize=(12, 12))

    color_list = ["blue", "red"]
    for i, score_data in enumerate(score_data_list):
        score_weights = np.ones_like(score_data) / len(score_data)
        ax.hist(
            score_data,
            alpha=0.5,
            color=color_list[i],
            label=label_list[i],
            weights=score_weights,
        )
    ax.set_xlabel("Value")
    ax.set_ylabel("Normalized Frequency")
    ax.set_title(f"{eval_column}")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{eval_column}_hist.png")
    plt.close()


def calculate_metrics(data_list, labels, eval_column, save_dir):

    # ── ROC‑AUC ──
    fpr, tpr, roc_th = roc_curve(labels, data_list)
    auc = roc_auc_score(labels, data_list)
    print(f"ROC AUC = {auc:.4f}")

    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(f"{save_dir}/{eval_column}_roc_curve.png")
    plt.close()

    # ── PR‑AP ──
    precision, recall, pr_th = precision_recall_curve(labels, data_list)
    ap = average_precision_score(labels, data_list)
    print(f"Average Precision = {ap:.4f}")

    plt.figure(figsize=(10, 10))
    plt.plot(recall, precision, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(f"{save_dir}/{eval_column}_pr_curve.png")
    plt.close()

    return auc, ap


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path2dataset", type=str, default="dataset/classification", required=True
    )
    parser.add_argument(
        "--pred_dir", type=str, default="results/0619_debug", required=True
    )
    parser.add_argument("--eval_column", type=str, default="crs", required=True)
    args = parser.parse_args()
    return args


def run_main():
    args = get_args()
    path2dataset = args.path2dataset
    pred_dir = args.pred_dir
    eval_column = args.eval_column
    dataset_name = path2dataset.split("/")[-1]
    save_dir = f"{pred_dir}/metric_results/{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    path2pred = f"{pred_dir}/pred_data/{dataset_name}_pred_data.json"

    print("DATASET PATH:", path2dataset)
    print("PRED PATH:", path2pred)
    print("EVAL COLUMN:", eval_column)
    print("SAVE DIR:", save_dir)
    main(
        save_dir,
        path2dataset,
        path2pred,
        eval_column,
    )


if __name__ == "__main__":
    run_main()
