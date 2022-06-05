import torchvision.transforms


transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize(480),
    torchvision.transforms.RandomResizedCrop(size=(384, 380)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
])

transform_eval = torchvision.transforms.Compose([
    torchvision.transforms.Resize(480),
    torchvision.transforms.RandomResizedCrop(size=(384, 380)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
])
