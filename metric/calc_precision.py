import json
import os
import argparse
import glob


def calc_precision(pred_list, gt_list, consider_index_list=None):
    precision = 0

    if consider_index_list is not None:
        total_num = len(consider_index_list)
    else:
        total_num = len(pred_list)

    for i, pred in enumerate(pred_list):
        if consider_index_list is not None and i not in consider_index_list:
            continue
        if pred == gt_list[i]:
            precision += 1
    precision /= total_num
    return precision


def convert_to_pred_list(path2pred, eval_column):
    with open(path2pred, "r") as f:
        data = json.load(f)
    data = dict(sorted(data.items(), key=lambda x: int(x[0])))
    pred_list = []
    for set_id, item in data.items():
        eval_scores = item[eval_column]
        eval_pred = eval_scores.index(max(eval_scores))
        pred_list.append(eval_pred)
    return pred_list


def convert_to_gt_list(path2dataset):
    gt_file_list = sorted(glob.glob(os.path.join(path2dataset, "*.json")))
    gt_list_A = []
    gt_list_B = []
    same_index_list = []
    different_index_list = []
    for i, gt_file in enumerate(gt_file_list):
        with open(gt_file, "r") as f:
            data = json.load(f)
        gt_list_A.append(data["Judgement"]["GT_A"])
        gt_list_B.append(data["Judgement"]["GT_B"])
        if data["Judgement"]["GT_same"]:
            same_index_list.append(i)
        else:
            different_index_list.append(i)

    return gt_list_A, gt_list_B, same_index_list, different_index_list


def main(
    save_dir,
    path2dataset,
    path2pred,
    eval_column,
):
    pred_list = convert_to_pred_list(path2pred, eval_column)
    gt_list_A, gt_list_B, same_index_list, different_index_list = convert_to_gt_list(
        path2dataset
    )
    assert (
        len(pred_list)
        == len(gt_list_A)
        == len(gt_list_B)
        == len(same_index_list) + len(different_index_list)
    )

    precision_A = calc_precision(pred_list, gt_list_A, None)
    precision_B = calc_precision(pred_list, gt_list_B, None)
    precision_same = calc_precision(pred_list, gt_list_A, same_index_list)
    precision_diff_A = calc_precision(pred_list, gt_list_A, different_index_list)
    precision_diff_B = calc_precision(pred_list, gt_list_B, different_index_list)

    stats_dict_1 = {
        "precision_A": precision_A,
        "precision_B": precision_B,
    }
    stats_dict_2 = {
        "precision_same": precision_same,
    }
    stats_dict_3 = {
        "precision_diff_A": precision_diff_A,
        "precision_diff_B": precision_diff_B,
    }

    num_all = len(pred_list)
    num_same = len(same_index_list)
    num_diff = len(different_index_list)

    print("----------------------------------------------")
    print("num_of_data", num_all)
    print_aligned_stats(stats_dict_1)
    print("----------------------------------------------")
    print("num_of_data_same", num_same, f"({num_same/num_all*100:.2f}%)")
    print_aligned_stats(stats_dict_2)
    print("----------------------------------------------")
    print("num_of_data_diff", num_diff, f"({num_diff/num_all*100:.2f}%)")
    print_aligned_stats(stats_dict_3)
    print("----------------------------------------------")

    save_result(
        num_all,
        num_same,
        num_diff,
        precision_A,
        precision_B,
        precision_same,
        precision_diff_A,
        precision_diff_B,
        save_dir,
        eval_column,
    )


def print_aligned_stats(stats_dict):
    max_key_length = max(len(str(k)) for k in stats_dict.keys())
    max_value_length = max(len(f"{v*100:.2f}") for v in stats_dict.values())
    format_str = f"{{:<{max(max_key_length, 35)}}}: {{:>{max_value_length}.2f}}%"
    for key, value in stats_dict.items():
        print(format_str.format(key, value * 100))


def save_result(
    num_all,
    num_same,
    num_diff,
    precision_A,
    precision_B,
    precision_same,
    precision_diff_A,
    precision_diff_B,
    save_dir,
    eval_column,
):
    with open(os.path.join(save_dir, f"{eval_column}_stats.txt"), "w") as f:
        f.write(f"num_all: {num_all}\n")
        f.write(f"num_same: {num_same}\n")
        f.write(f"num_diff: {num_diff}\n")
        f.write(f"precision_A: {precision_A:.4f}\n")
        f.write(f"precision_B: {precision_B:.4f}\n")
        f.write(f"precision_same: {precision_same:.4f}\n")
        f.write(f"precision_diff_A: {precision_diff_A:.4f}\n")
        f.write(f"precision_diff_B: {precision_diff_B:.4f}\n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path2dataset", type=str, default="dataset/pairwise_comparison", required=True
    )
    parser.add_argument("--pred_dir", type=str, default="results/demo", required=True)
    parser.add_argument("--eval_column", type=str, default="crs_map", required=True)
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
    main(
        save_dir,
        path2dataset,
        path2pred,
        eval_column,
    )


if __name__ == "__main__":
    run_main()
