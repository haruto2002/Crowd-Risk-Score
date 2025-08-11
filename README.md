# Crowd Risk Score (CRS) 

## Clone Repository
```bash
git clone git@github.com:haruto2002/Crowd-Risk-Score.git
cd Crowd_Risk_Score
```

## 仮想環境の構築
```bash
conda create -n crs python=3.10
conda activate crs
pip install numpy scipy opencv-python tqdm pyyaml matplotlib scikit-learn
```

## 使用方法

### データの準備

以下のディレクトリ構造が必要です：

```
trajectory_data/
└── WorldPorter_202408_0001/
    ├── track_frame_data/
    │   ├── 0001.txt
    │   ├── 0002.txt
    │   └── ...
    ├── homography_matrix.txt
    └── map_size.txt
```

### プログラムの実行

#### Trajectory data に対するCrowd Risk Scoreの計算

##### コマンドラインで変数を指定する場合
```bash
python src/main.py \
    --results_base_dir_name results \
    --dir_name demo \
    --trajectory_dir trajectory_data/WorldPorter_202408_0001 \
    --grid_size 5 \
    --vec_span 10 \
    --freq 10 \
    --R 13.5 \
    --frame_start 1 \
    --frame_end 8990
```

##### YAML設定ファイルを使用する場合

`src/config/config.yaml` ファイルを編集して、必要なパラメータを設定します：

```yaml
results_base_dir_name: results
dir_name: 0811_debug
trajectory_dir: trajectory_data/WorldPorter_202408_0001
crop_area: null
frame_range:
- 1
- 8990
freq: 10
R: 13.5
grid_size: 5
vec_span: 10
```

```bash
python src/main.py --use_yaml --yaml_path src/config/config.yaml
```

#### データセットに対応したスコアの算出

```bash
# Pairwise comparison dataset
python metric/set_prediction.py --path2dataset dataset/pairwise_comparison --dataset_type pairwise_comparison --pred_dir results/demo

# Classification dataset
python metric/set_prediction.py --path2dataset dataset/classification --dataset_type classification --pred_dir results/demo
```

#### 評価指標の計算
```bash
# Pairwise comparison dataset
python metric/calc_metric.py --path2dataset dataset/pairwise_comparison --dataset_type pairwise_comparison --pred_dir results/demo --eval_column crs

# Classification dataset
python metric/calc_metric.py --path2dataset dataset/classification --dataset_type classification --pred_dir results/demo --eval_column crs
```

## パラメータの説明

- `results_base_dir_name`: 結果保存のベースディレクトリ名
- `dir_name`: 出力ディレクトリ名
- `trajectory_dir`: 軌道データのディレクトリパス
- `crop_area`: クロップエリア（nullの場合は全体）
- `frame_range`: 処理するフレーム範囲 [開始, 終了]
- `freq`: 危険度計算のフレーム間隔
- `R`: ガウシアンカーネルの半径パラメータ
- `grid_size`: グリッドサイズ
- `vec_span`: ベクトル計算のスパン

## 出力

実行後、以下のディレクトリが作成されます：

```
results/0811_debug/
├── each_result/
│   ├── crs_map/
│   │   ├── 0001_0011.txt
│   │   ├── 0011_0021.txt
│   │   └── ...
│   ├── vec_data/
│   │   ├── 0001_0011.txt
│   │   ├── 0011_0021.txt
│   │   └── ...
│   └── config.yaml
├── pred_data/
└── metric_results/
```


## プロジェクト構造

```
Crowd_Risk_Score/
├── src/
│   ├── main.py              # メイン実行ファイル
│   ├── config/
│   │   └── config.yaml      # 設定ファイル
│   └── utils/
│       ├── clac_crowd_risk_score.py  # crs計算
│       └── get_track_data.py         # 軌道データ処理
├── dataset/
│   ├── pairwise_comparison/
│   └── classification/
├── metric/                   # 評価指標計算
│   ├── calc_classification_scores.py  # 分類スコア計算
│   ├── calc_metric.py                 # メトリクス計算
│   ├── set_classification_pred.py     # 分類予測設定
│   ├── set_prediction.py             # 予測設定
│   ├── set_pairwise_pred.py          # ペアワイズ予測設定
│   ├── calc_precision.py             # 精度計算
│   └── utils.py                      # ユーティリティ関数
├── results/                 # 結果出力
├── trajectory_data/         # 軌道データ
└── README.md
```
