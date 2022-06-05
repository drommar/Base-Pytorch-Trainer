import torchvision
import torch.nn as nn
from model.submodules import output_net


def better_model():
    finetune_net = nn.Sequential()
    backbone_net = torchvision.models.resnet152(pretrained=True)

    finetune_net.backbone_net = backbone_net
    finetune_net.output_net = output_net()

    for param in finetune_net.backbone_net.parameters():
        param.requires_grad = False

    for param in finetune_net.backbone_net.fc.parameters():
        param.requires_grad = True

    for param in finetune_net.backbone_net.avgpool.parameters():
        param.requires_grad = True

    for param in finetune_net.backbone_net.layer4.parameters():
        param.requires_grad = True

    # for param in finetune_net.backbone_net.layer3.parameters():
    #     param.requires_grad = True
    #
    # for param in finetune_net.backbone_net.layer2.parameters():
    #     param.requires_grad = True
    #
    # for param in finetune_net.backbone_net.layer1.parameters():
    #     param.requires_grad = True

    return finetune_net


def main():
    my_net = better_model()
    print(my_net)
    pass


if __name__ == '__main__':
    main()
