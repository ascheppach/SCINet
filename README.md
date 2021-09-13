### Introduction: SCINet
First implementation of following paper: [*Time Series is a Special Sequence: Forecasting with Sample Convolution and Interaction*](https://arxiv.org/abs/2106.09305v1)

The goal of the newly proposed neural network architecture named Sample Convolution and Interaction Network (SCINet) is facing the challenge of handling time series data, while keeping their unique properties. State-of-the-art models for time series forecasting include recurrent neural networks and Transformer Models. Both come with disadvantages: while RNN-based methods suffer from error accumulation, Transformers can hardly be applied to long-sequence time series forecasting due to quadratic time complexity, high memory usage, and inherent limitation of the encoder-decoder architecture. The authors propose to use temporal convolution models. Using these, a fast data processing and efficient dependencies learning is made possible by allowing the parallel convolution operation of multiple filters. While other state-of-the-art methods for time series forecasting such as WaveNets include dilated causal convolution, SCINet is designed to lift its constraints, for instance the restricted possibility of extraction of temporal dynamics from the data/feature in the previous layer, and the necessity of having a network with equal input and output length. Furthermore, the authors assume that time series data can be downsampled without losing information, which is not the case for general sequence data such as texts or DNA. This property is used within the SCINet framework, for instance to save computational resources and thus, to train the model more efficiently, which they prove within their experiments.

## Repository structure
Our Repository consists of the main folder, containing our general functions such as the data preprocessing and the training script, as well as the script for evaluating the model performance. Furthermore, there are two folders inside here:
- The **data** folder consists of the downloaded dataset files on [electricity](https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip) and [traffic](https://github.com/laiguokun/multivariate-time-series-data/blob/7f402f185cc2435b5e66aed13a3b560ed142e023/traffic/traffic.txt.gz) as stated in the mentioned paper and store them in here. 
- The **models** folder consists of our data preprocessing, and one baseline model for reference, as well as the SCINet implementation, including all three building blocks as described in the paper.

The main file is the `train.py`, which sources all other relevant files for running the model. 

## Setup
```
conda create --name SCINet python=3.6

source activate SCINet

git clone https://github.com/ascheppach/SCINet.git

cd SCINet

export PYTHONPATH="$PYTHONPATH:~/SCINet"
```

## Run SCINet for time series
To run SCINet for time series, enter the command below. Please note, that here, the relative data_directory is used. If you receive an error message, please check if your current directory is the pulled project repository folder.

```
# traffic
python train.py --model='SCINet' --model_path='./run1_traffic.pth' --epochs=100 --num_steps=30000 --horizon=3 --batch_size=16 --seq_size=168 --learning_rate=0.0005 --k=5 --num_motifs=320 --h=2  --L=3 --padding=4 --seed=4321 --data_directory=data\traffic.txt --data_name='traffic'

# electricity
python train.py --model='SCINet' --model_path='./run1_elec.pth' --epochs=10 --num_steps=6000 --horizon=3 --batch_size=32 --seq_size=168 --learning_rate=0.0005 --k=5 --num_motifs=320 --h=8 --K=1 --L=3 --padding=4 --seed=4321 --data_directory=data\LD2011_2014.txt --data_name='electricity'
```

For the settings stated in the command above, following performances were achieved for traffic with SCInet (first plot), for traffic with SCInet2 (second plot) and electricity with SCInet (third plot):

![](/results/train_val-loss_traffic_new_scinet.jpg)
![](/results/train_val-loss_traffic_new_scinet2.jpg)
![](/results/train_val-loss_electricity.jpg)

## Tested with

- Python 3.6
- Ubuntu 20.04
- torch 1.7.1+cu101
