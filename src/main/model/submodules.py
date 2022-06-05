import torch.nn as nn


def conv2d_bn_relu(in_dim, out_dim, kernel, stride=1, pad=0, dilate=1, group=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel, stride, pad, dilate, group),
        nn.BatchNorm2d(out_dim),
        nn.ReLU()
    )


def up_conv2d(in_dim, out_dim, kernel=3, pad=1, up_scale=2):
    return nn.Sequential(
        nn.Upsample(scale_factor=up_scale, mode='nearest'),
        nn.Conv2d(in_dim, out_dim, kernel, padding=pad)
    )


def output_net():
    output_layers = nn.Sequential(
        nn.Linear(in_features=1000, out_features=512, bias=True),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(in_features=512, out_features=256, bias=True),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=128, bias=True),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=10, bias=True)
    )
    return output_layers
