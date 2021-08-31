### Introduction: SCINet
First implementation of following paper: *Time Series is a Special Sequence: Forecasting with Sample Convolution and Interaction*

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

## Repository structure
Our Repository consists of the main folder, containing our general functions such as the data preprocessing and the training scripts. Furthermore, there are two folders:
- The data folder is empty, as the files are too large to include in our repository. To get the datasets, download the files on [electricity](https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip) and [traffic](https://github.com/laiguokun/multivariate-time-series-data/blob/7f402f185cc2435b5e66aed13a3b560ed142e023/traffic/traffic.txt.gz) as stated in the mentioned paper and store them in here. 
- The models folder consists of our data preprocessing, and the SCINet implementation, as well as one baseline model for reference.

The main file is the `train.py`, which sources all other relevant files for running the model. 

## Run SCINet for time series
To run SCINet for time series, enter the command below. Please note, that here, the relative data_directory is used. If you receive an error message, please check if your current directory is the pulled project repository folder.

```
python train.py --model_path='./run1.pth' --epochs=100 --num_steps=3000 --horizon=3 --batch_size=16 --seq_size=168 --learning_rate=0.0005 --k=5 --num_motifs=320 --h=2 --K=1 --L=3 --padding=4 --seed=4321 --data_directory=data\traffic.txt
```

![](/train_val-loss.jpg)

