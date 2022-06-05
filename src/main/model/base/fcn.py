import torchvision
import torch.nn as nn
from model.submodules import output_net


def base_model():
    finetune_net = nn.Sequential()
    backbone_net = torchvision.models.resnet50(pretrained=True)

    finetune_net.backbone_net = backbone_net
    finetune_net.output_net = output_net()

    for param in finetune_net.backbone_net.parameters():
        param.requires_grad = False

    for param in finetune_net.backbone_net.fc.parameters():
        param.requires_grad = True

    return finetune_net


def main():
    pass


if __name__ == '__main__':
    main()
