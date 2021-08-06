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


## Run ResNet for time series
To run ResNet for time series, enter following command (change data_directory to your corresponding data path)

```
python train.py --epochs=10 --k=2 --padding=1 --num_steps=100 --batch_size=2 --data_directory=data/traffic.txt
```
