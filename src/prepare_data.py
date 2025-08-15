import torch
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision.datasets import CIFAR10


def prepare_data():

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010],
            ),
        ]
    )

    full_train_ds = CIFAR10(
        root="../dataset/train", train=True, transform=transform, download=True
    )
    test_ds = CIFAR10(
        root="../dataset/test",
        train=False,
        transform=transform,
        download=True,
    )

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, val_indices = next(
        sss.split(full_train_ds.data, full_train_ds.targets)
    )

    train_ds = torch.utils.data.Subset(full_train_ds, train_indices)
    val_ds = torch.utils.data.Subset(full_train_ds, val_indices)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=128, shuffle=True, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=128, shuffle=False, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=128, shuffle=False, num_workers=4
    )
    return train_loader, val_loader, test_loader
