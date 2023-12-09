# CARLA Dataset Offline Training Repository

## Overview
This repository is dedicated to training models using offline datasets from the CARLA Simulator. It is designed to support various deep learning tasks including VAE, ResNet, and LSTM models on BEV (Bird's Eye View) and FPV (First Person View) images.

### Supported Models and Loss Functions (See details in our paper)
- **VAE on BEV Images**: Training Variational Autoencoders (VAE) on BEV images with a reconstruction loss.
- **ResNet Finetuning on FPVBEV Dataset**: Finetuning ResNet on the FPVBEV dataset using a contrastive loss.
- **LSTM on Time-Series BEV-Action Dataset**: Training Long Short-Term Memory (LSTM) on time-series BEV-Action dataset.

## Getting Started

### Prerequisites
- Dataset collected yourself or download from here. 
- Python 3.x
- Necessary Python libraries (listed in `requirements.txt`)

### Installation
Clone this repository.

### Dataset
Before training, you need to configure the dataset path:
- Modify the '**root_dir**' in '**initial.py**' to point to the location of your dataset.
- Use the '**--expname DIRECTORY_NAME**' argument to specify the dataset's directory.
Your dataset should be structured as follows:
- '**fpv.npy**': FPV image dataset.
- '**bev.npy**': BEV image dataset.
- '**action.npy**': Action value dataset.
- '**terminal.npy**': A binary array file indicating the terminal moment for each episode.

### Training Models
To train a model, run the '**train.py**' script. Use the '**-h**' option to see all available training options.
Use the '**--model**' option to select the model you want to train:
- **BEV_VAE_CARLA**: Training the VAE model.
- **FPV_BEV_CARLA**: Finetuning ResNet model.
- **BEV_LSTM_CARLA**: Training the LSTM model.

Sample Usage:
1. Training the VAE model. '**SAVE_DIR**' is where your checkpoints are saved.'**MODEL_PATH**' is the name of the checkpoint file. '**EXP_NAME**' is where your dataset is stored.
```
python train.py --model BEV_VAE_CARLA --load_checkpoint --save_dir \[SAVE_DIR\] --model_path \[MODEL_PATH\] --expname \[EXP_NAME\] --nepoch 100 --train_batch_size 256 --sample_batch_size 64
```
2. Finetuning ResNet model. Change the '**vae_model_path**' in '**initial.py**' to point to the location of your VAE checkpoint.
```
python train.py --model FPV_BEV_CARLA --load_checkpoint --save_dir \[SAVE_DIR\] --model_path \[MODEL_PATH\] --expname \[EXP_NAME\] --nepoch 100 --train_batch_size 128 --sample_batch_size 512
```
3. Training the LSTM model.
```
python train.py --model BEV_LSTM_CARLA --load_checkpoint --save_dir \[SAVE_DIR\] --model_path \[MODEL_PATH\] --expname \[EXP_NAME\] --nepoch 100 --train_batch_size 1 --sample_batch_size 512
```
