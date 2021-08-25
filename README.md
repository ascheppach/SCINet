### Introduction: SCINet
First implementation of following paper: Time Series is a Special Sequence: Forecasting with Sample Convolution and Interaction

## Tested with

- Python 3.6
- Ubuntu 20.04
- torch 1.7.1+cu101

## Setup
```
conda create --name SCINet python=3.6

source activate SCINet

git clone https://github.com/ascheppach/SCINet.git

cd SCINet

export PYTHONPATH="$PYTHONPATH:~/SCINet"
```

## Data
To get the data, download the zip file and store them in /data folder.

## Run SCINet for time series
To run SCINet for time series, enter following command (relative data_directory used, if error: check if your cd is the project repo folder)

```
python train.py --model_path='./run1.pth' --epochs=100 --num_steps=100 --horizon=3 --batch_size=16 --seq_size=168 --learning_rate=0.0005 --k=5 --num_motifs=320 --h=2 --K=1 --L=3 --padding=4 --seed=4321 --data_directory=data\traffic.txt
```
