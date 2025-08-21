# Crowd Risk Score (CRS) 
![overview](figs/overview.gif)

## Clone Repository
```bash
git clone git@github.com:haruto2002/Crowd-Risk-Score.git
cd Crowd_Risk_Score
```

## Virtual Environment Setup
```bash
conda create -n crs python=3.10
conda activate crs
pip install numpy scipy opencv-python tqdm pyyaml matplotlib scikit-learn
```

## Usage

### Data Preparation

#### Trajectory Data

You can download the trajectory data from Google Drive:
[**Trajectory Data**](https://drive.google.com/drive/folders/1WkfkgLNH09XLlwDKt8zdqA5HeWCzP9rx?usp=sharing)

The following directory structure is required:
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

#### Evaluation Dataset

You can download the datasets from Google Drive:
[**Dataset**](https://drive.google.com/drive/folders/1qlmEkQEn4RpqOGX4hrKXRy4iFzf7YiC_?usp=sharing)

The following directory structure is required:
```
dataset/
└── WorldPorter_202408_0001/
    ├── classification/
    │   ├── 0001.json
    │   ├── 0002.json
    │   └── ...
    └── pairwise_comparison/
        ├── 0001.json
        ├── 0002.json
        └── ...
```

##### Classification Dataset
```
  "frame_range": Target frame range in trajectory_data >> [start_frame, end_frame]
  "crop_points": Target area in trajectory_data >> [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
  "GT_A": Annotator A's classification judgment >> 0(Safe) or 1(Danger)
  "GT_B": Annotator B's classification judgment >> 0(Safe) or 1(Danger)
  "GT_same": Agreement/disagreement between two annotators >> True or False
```
##### Pairwise Comparison Dataset
```
  "1", "2": Information for two scenes to be compared, each containing:
  - "frame_range": Target frame range in trajectory_data >> [start_frame, end_frame]  
  - "crop_points": Target area in trajectory_data >> [top_left_x, top_left_y, bottom_right_x, bottom_right_y]  

  "Judgement":  
  - "GT_A": Annotator A's comparison judgment >> 1("1" is higher risk) or 2("2" is higher risk)  
  - "GT_B": Annotator B's comparison judgment >> 1("1" is higher risk) or 2("2" is higher risk)   
  - "GT_same": Agreement/disagreement between two annotators >> True or False
```

### Program Execution

#### Calculating Crowd Risk Score for Trajectory Data

##### Specifying variables via command line
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

##### Using YAML configuration file

Edit the `src/config/config.yaml` file to set the required parameters:

```yaml
results_base_dir_name: results
dir_name: demo
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

##### Parameter Description

- `results_base_dir_name`: Base directory name for saving results
- `dir_name`: Output directory name
- `trajectory_dir`: Directory path for trajectory data
- `crop_area`: Crop area (null for entire area)
- `frame_range`: Frame range to process [start, end]
- `freq`: Frame interval for risk calculation
- `R`: Radius parameter for Gaussian kernel
- `grid_size`: Grid size
- `vec_span`: Vector calculation span

#### Calculating scores corresponding to the dataset

```bash
# Pairwise comparison dataset
python metric/set_prediction.py --path2dataset dataset/WorldPorter_202408_0001/pairwise_comparison --dataset_type pairwise_comparison --pred_dir results/demo

# Classification dataset
python metric/set_prediction.py --path2dataset dataset/WorldPorter_202408_0001/classification --dataset_type classification --pred_dir results/demo
```

#### Calculating evaluation metrics
```bash
# Pairwise comparison dataset
python metric/calc_metric.py --path2dataset dataset/WorldPorter_202408_0001/pairwise_comparison --dataset_type pairwise_comparison --pred_dir results/demo --eval_column crs

# Classification dataset
python metric/calc_metric.py --path2dataset dataset/WorldPorter_202408_0001/classification --dataset_type classification --pred_dir results/demo --eval_column crs
```

## Output

After execution, the following directory will be created:

```
results/demo/
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


## Project Structure

```
Crowd_Risk_Score/
├── src/
│   ├── main.py              # Main execution file
│   ├── config/
│   │   └── config.yaml      # Configuration file
│   └── utils/
│       ├── clac_crowd_risk_score.py  # CRS calculation
│       └── get_track_data.py         # Trajectory data processing
├── dataset/
│   └──WorldPorter_202408_0001/
│       ├── classification/
│       └── pairwise_comparison/
├── metric/                   # Evaluation metric calculation
│   ├── calc_classification_scores.py  # Classification score calculation
│   ├── calc_metric.py                 # Metric calculation
│   ├── set_classification_pred.py     # Classification prediction setup
│   ├── set_prediction.py             # Prediction setup
│   ├── set_pairwise_pred.py          # Pairwise prediction setup
│   ├── calc_precision.py             # Precision calculation
│   └── utils.py                      # Utility functions
├── results/                 # Result output
├── trajectory_data/         # Trajectory data
└── README.md
```
