import torchvision
import torch.nn as nn


def efficient_model():
    finetune_net = torchvision.models.efficientnet_b4(pretrained=True)

    for i, param in enumerate(finetune_net.parameters()):
        if i > 126:
            param.requires_grad = True
        else:
            param.requires_grad = False

    finetune_net.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features=1792, out_features=1000, bias=True),
        nn.Dropout(p=0.15, inplace=True),
        nn.Linear(in_features=1000, out_features=10, bias=True)
    )

    return finetune_net


def main():
    my_net = efficient_model()
    print(my_net)
    for i, name_param in enumerate(my_net.named_parameters()):
        print(i, name_param)


if __name__ == '__main__':
    main()
