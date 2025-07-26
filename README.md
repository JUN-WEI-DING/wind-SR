
# Wind Super-Resolution Forecasting

This repository contains code for super-resolution of wind fields using GAN-based models. The project includes data preparation, model training, inference, and performance evaluation using multiple data sources including ICON, ERA5, ground stations, and wind farm data.

## 📁 Folder Structure

```
.
├── data/                      # Contains Zarr and CSV wind data (Currently not available due to manuscript under review.)
├── gan/                      # GAN model training and evaluation logic
├── model/                    # Model architecture (Generator, Critic)
├── model_checkpoiints/      # Training checkpoints
├── model_parameter/         # Inference model weights
├── utils/                    # Utility functions (normalization, IO, etc.)
├── ground_station_evaluation.py   # Evaluation using ground station observations
├── inference.py             # Script for GAN inference and SR image generation
├── train.py                 # Model training entry point
├── wind_power_evaluation.py # Wind farm power estimation and evaluation
├── requirements.txt         # Python dependencies
└── README.md                # Project overview (this file)
```

## 🚀 Getting Started


### 1. Setup

```bash
git clone https://github.com/JUN-WEI-DING/wind-SR.git
cd wind-sr-forecasting
pip install -r requirements.txt
```

### 2. Train Model

Train the GAN model using ICON/ERA5 data:

```bash
python train.py --z_dim 1 --input_channel 8 --output_channel 2 --N 3 --batch_size 256 --epochs 500
```

📝 All training parameters (e.g., `--lr`, `--clip_value`, `--model_save`, etc.) can be customized via command-line arguments.

### 3. Run Inference

Generate super-resolved wind maps:

```bash
python inference.py --z_dim 1 --input_channel 8 --output_channel 2 --N 3
```

📝 You may also configure number of samples, device, or batch size using additional arguments.

### 4. Ground Station Evaluation

Evaluate model accuracy against real-world ground station measurements:

```bash
python ground_station_evaluation.py
```

### 5. Wind Farm Power Estimation

Simulate wind power generation using predicted wind fields:

```bash
python wind_power_evaluation.py
```

📝 All scripts support optional arguments for flexible model configuration, data paths, and evaluation settings.


## 🧠 Model Overview

- **Generator**: Conditional convolutional GAN for spatial super-resolution.
- **Critic**: Discriminator model using multi-scale patches.
- **Input**: ERA5 low-resolution wind fields.
- **Target**: Super-Resolution Wind Farm.

## 📊 Data Sources

- ICON Zarr dataset
- ERA5 Zarr dataset
- Ground station observations
- Wind farm CSV 

## 📦 Requirements

See `requirements.txt` for dependencies. Key packages include:

- `torch`
- `xarray`
- `cartopy`
- `matplotlib`
- `numpy`
- `pandas`

## 👤 Author

Developed by Jun-Wei Ding, NTU Civil Engineering  
Contact: d13521023@ntu.edu.tw
