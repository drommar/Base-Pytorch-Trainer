# Base-Pytorch-Trainer

## Project structure

```
.
├── configs: Put configure files
├── dataset: Put dataset files
├── submissions: Put submission files
├── src
│   └── main
│       ├── data: Put data-processing code
│       ├── engine: Put trainer, predicter, and splitter code
│       ├── model: Put model code
│       ├── options: Put argparse code
│       ├── predict.py: Predict entry
│       ├── split_dataset.py: Split dataset entry
│       └── train.py: Train entry
├── do_predict.sh
├── do_split_dataset.sh
├── do_tensorboard.sh
└── do_train.sh
```

## Tips

If you are using pycharm, set `src/main` as sources root.

## How to use

### 1, Install requirements

`pip install tensorboard`

`conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch pyyaml`

### 2, Put your dataset to `./dataset`

A generic dataset folder where the images are arranged in this way by default: 
```
root/train_images/label1/xxx.png
root/train_images/label1/xxx.png
...
root/train_images/label2/xxx.png
root/train_images/label2/xxx.png
...
...
root/test_images/xxy.png
root/test_images/xxy.png
...
```

### 3, Set your parameter

You can set parameter in `config/xxx.yaml` which will be set as default parameter.

Then, you can override any parameter in shell script.

### 4, Split dataset

Running Script `./do_split_dataset.sh`

### 5, Create model

Create your model in `src/main/model`

### 6, Train

Running Script `./do_train.sh`

### 7, Predict

Running Script `./do_predict.sh`

Every row of submission file should already have ID
