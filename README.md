# ClimateBench-MindSpore

A MindSpore framework-port reproduction of the ClimateBench CNN-LSTM climate emulator, developed and tested on Huawei Cloud ModelArts.

## 1. Task

The model learns a mapping from climate forcing inputs to annual global climate-response fields.

### Inputs

| Variable | Meaning | Representation |
|---|---|---|
| `CO2` | carbon dioxide concentration | global time series |
| `CH4` | methane concentration | global time series |
| `SO2` | sulfur-dioxide/aerosol forcing | spatiotemporal field |
| `BC` | black-carbon forcing | spatiotemporal field |

### Targets

| Variable | Meaning |
|---|---|
| `tas` | near-surface air temperature |
| `pr` | precipitation |
| `diurnal_temperature_range` | diurnal temperature range |
| `pr90` | 90th-percentile precipitation / extreme precipitation |

The target grid is `96 × 144` latitude-longitude points.

## 2. Experiment protocol

Training experiments:

```text
historical, ssp126, ssp370, ssp585, hist-GHG, hist-aer
```

Evaluation experiment:

```text
ssp245
```

Temporal input window:

```text
10 years
```

Model input and output shapes:

```text
input:  [batch, 10, 96, 144, 4]
output: [batch, 1, 96, 144]
```

## 3. Environment

The reported runs used:

| Item | Value |
|---|---|
| Platform | Huawei Cloud ModelArts Notebook |
| GPU | NVIDIA Tesla T4 |
| Framework | MindSpore GPU 1.7.0 |
| Python | 3.7.10 |
| Persistent workspace | `/home/ma-user/work` |

The ModelArts image provides MindSpore. Install the remaining Python dependencies with:

```bash
/home/ma-user/anaconda3/envs/MindSpore/bin/python -m pip install -r requirements.txt
```

## 4. Repository structure

The lightweight repository package contains:

```text
ClimateBench-MindSpore/
├── .gitignore
├── README.md
├── requirements.txt
├── src/
│   ├── data_utils.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   ├── metrics.py
│   ├── metrics_official.py
│   └── plot_maps.py
└── outputs/
    ├── checkpoints/
    │   ├── stats_tas.json
    │   ├── stats_pr.json
    │   ├── stats_diurnal_temperature_range.json
    │   ├── stats_pr90.json
    │   └── train_log_*.csv
    └── figures/
        └── *_ssp245_{2030,2050,2100}_{truth,pred,error}.png
```

## 5. Data preparation

Place the processed ClimateBench files in `data/`:

```text
inputs_historical.nc
outputs_historical.nc
inputs_ssp126.nc
outputs_ssp126.nc
inputs_ssp370.nc
outputs_ssp370.nc
inputs_ssp585.nc
outputs_ssp585.nc
inputs_hist-GHG.nc
outputs_hist-GHG.nc
inputs_hist-aer.nc
outputs_hist-aer.nc
inputs_ssp245.nc
outputs_ssp245.nc
```

The processed data are available from the ClimateBench Zenodo record linked in the references.

## 6. Model

The MindSpore model follows the main CNN-LSTM layout of the ClimateBench neural-network baseline:

```text
Conv2D(4 → 20, 3×3)
→ ReLU
→ AveragePooling2D
→ global spatial mean
→ LSTM(hidden size 25)
→ Dense(25 → 96×144)
→ reshape to 1×96×144
```

Implementation: `src/model.py`.

## 7. Training

Train one target variable:

```bash
cd /home/ma-user/work/ClimateBench-MindSpore

/home/ma-user/anaconda3/envs/MindSpore/bin/python src/train.py \
  --data_dir data \
  --var tas \
  --epochs 30 \
  --batch_size 16 \
  --device_target GPU
```

Replace `tas` with `pr`, `diurnal_temperature_range`, or `pr90`.

## 8. Prediction

```bash
/home/ma-user/anaconda3/envs/MindSpore/bin/python src/predict.py \
  --data_dir data \
  --var tas \
  --scenario ssp245 \
  --ckpt outputs/checkpoints/cnn_lstm_tas_final.ckpt \
  --stats outputs/checkpoints/stats_tas.json \
  --batch_size 16 \
  --device_target GPU
```

## 9. Evaluation

Full-period diagnostic evaluation:

```bash
/home/ma-user/anaconda3/envs/MindSpore/bin/python src/metrics.py \
  --data_dir data \
  --var tas \
  --scenario ssp245 \
  --pred_path outputs/predictions/pred_ssp245_tas.nc
```

ClimateBench 2080–2100 evaluation:

```bash
/home/ma-user/anaconda3/envs/MindSpore/bin/python src/metrics_official.py \
  --data_dir data \
  --var tas \
  --scenario ssp245 \
  --pred_path outputs/predictions/pred_ssp245_tas.nc \
  --year_start 2080 \
  --year_end 2100
```

## 10. Visualization

```bash
/home/ma-user/anaconda3/envs/MindSpore/bin/python src/plot_maps.py \
  --data_dir data \
  --var tas \
  --scenario ssp245 \
  --pred_path outputs/predictions/pred_ssp245_tas.nc \
  --years 2030 2050 2100
```

For four variables, three years, and three map types, the expected total is 36 PNG files.

## 11. References

- ClimateBench paper: https://doi.org/10.1029/2021MS002954
- Official ClimateBench repository: https://github.com/duncanwp/ClimateBench
- Processed ClimateBench data: https://doi.org/10.5281/zenodo.5196512
- Huawei Cloud ModelArts: https://www.huaweicloud.com/intl/en-us/product/modelarts.html
