import argparse
from set_pairwise_pred import set_pairwise_pred
from set_classification_pred import set_classification_pred


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path2dataset", type=str, default="dataset/pairwise_comparison", required=True
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["pairwise_comparison", "classification"],
        required=True,
    )
    parser.add_argument(
        "--pred_dir", type=str, default="results/0619_debug", required=True
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    path2dataset = args.path2dataset
    dataset_type = args.dataset_type
    pred_dir = args.pred_dir
    if dataset_type == "pairwise_comparison":
        set_pairwise_pred(path2dataset, pred_dir)
    elif dataset_type == "classification":
        set_classification_pred(path2dataset, pred_dir)


if __name__ == "__main__":
    main()
