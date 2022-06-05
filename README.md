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
If you are using pycharm, set src/main as sources root.