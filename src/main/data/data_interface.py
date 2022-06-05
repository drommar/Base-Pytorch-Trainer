import os
import torchvision
from torch.utils.data import DataLoader
from data.augment import transform_train
from data.augment import transform_eval


def select_train_loader(args):
    train_dataset = torchvision.datasets.ImageFolder(os.path.join(args.dataset_dir, 'train_valid_test', "train"),
                                                     transform=transform_train)
    print(train_dataset.class_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                              drop_last=False)
    return train_loader


def select_eval_loader(args):
    eval_dataset = torchvision.datasets.ImageFolder(os.path.join(args.dataset_dir, 'train_valid_test', "valid"),
                                                    transform=transform_eval)
    val_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)
    return val_loader
