import argparse
import os
from calc_precision import main as run_precision
from calc_classification_scores import main as run_classification_scores


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path2dataset", type=str, default="dataset/pairwise_comparison", required=True
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["pairwise_comparison", "classification"],
        required=True
    )
    parser.add_argument(
        "--pred_dir", type=str, default="results/0619_debug", required=True
    )
    parser.add_argument(
        "--eval_column", type=str, default="crs", required=True
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    path2dataset = args.path2dataset
    dataset_type = args.dataset_type
    pred_dir = args.pred_dir
    eval_column = args.eval_column
    save_dir = f"{pred_dir}/metric_results/{path2dataset.split('/')[-1]}"
    os.makedirs(save_dir, exist_ok=True)
    path2pred = f"{pred_dir}/pred_data/{path2dataset.split('/')[-1]}_pred_data.json"

    print("DATASET NAME:", path2dataset.split('/')[-1])
    print("DATASET TYPE:", dataset_type)
    print("PRED PATH:", path2pred)
    print("EVAL COLUMN:", eval_column)
    print("SAVE DIR:", save_dir)
    if dataset_type == "pairwise_comparison":
        run_precision(
            save_dir,
            path2dataset,
            path2pred,
            eval_column,
        )
    elif dataset_type == "classification":
        run_classification_scores(
            save_dir,
            path2dataset,
            path2pred,
            eval_column,
        )


if __name__ == "__main__":
    main()






