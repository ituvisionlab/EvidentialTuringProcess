from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from torch.utils import data


def prepare_for_training_with_ood(train_dataset, ood_dataset):
    if len(train_dataset) < len(ood_dataset):
        id_ratio = np.ceil(float(len(ood_dataset)) / float(len(train_dataset)))
        assert id_ratio.is_integer()
        dataset_list = [
            train_dataset,
        ] * int(id_ratio)
        train_dataset = data.ConcatDataset(dataset_list)

    if len(train_dataset) > len(ood_dataset):
        ratio = np.ceil(float(len(train_dataset)) / float(len(ood_dataset)))
        assert ratio.is_integer()
        dataset_list = [
            ood_dataset,
        ] * int(ratio)
        ood_dataset = data.ConcatDataset(dataset_list)

        if len(ood_dataset) > len(train_dataset):
            ood_dataset = data.Subset(ood_dataset, np.arange(0, len(train_dataset)))

    assert len(train_dataset) == len(ood_dataset)

    return train_dataset, ood_dataset


def prepare_data_fashion(batch_size, iscuda):
    kwargs = {"num_workers": 1, "pin_memory": True} if iscuda else {}

    trafos = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.2860,), std=(0.3530,))
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    datadir = "~/Documents/data/"

    train_data = datasets.FashionMNIST(
        datadir + "fashion", train=True, download=True, transform=trafos
    )
    test_data = datasets.FashionMNIST(
        datadir + "fashion", train=False, download=True, transform=trafos
    )
    ood_data = datasets.MNIST(
        datadir + "mnist", train=True, download=True, transform=trafos
    )
    test_ood_data = datasets.MNIST(
        datadir + "mnist", train=False, download=True, transform=trafos
    )
    post_test_ood_data = datasets.KMNIST(
        datadir + "kmnist", train=False, download=True, transform=trafos
    )

    train_data, ood_data = prepare_for_training_with_ood(train_data, ood_data)
    test_data, test_ood_data = prepare_for_training_with_ood(test_data, test_ood_data)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )
    ood_loader = DataLoader(
        ood_data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )
    test_ood_loader = DataLoader(
        test_ood_data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )
    post_test_ood_loader = DataLoader(
        post_test_ood_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **kwargs
    )

    return train_loader, test_loader, ood_loader, test_ood_loader, post_test_ood_loader


def prepare_data_mnist(batch_size, iscuda):
    kwargs = {"num_workers": 1, "pin_memory": True} if iscuda else {}

    trafos = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))]
    )

    datadir = "~/Documents/data/"

    train_data = datasets.MNIST(
        datadir + "mnist", train=True, download=True, transform=trafos
    )
    test_data = datasets.MNIST(
        datadir + "mnist", train=False, download=True, transform=trafos
    )
    ood_data = datasets.FashionMNIST(
        datadir + "fashion", train=True, download=True, transform=trafos
    )
    test_ood_data = datasets.FashionMNIST(
        datadir + "fashion", train=False, download=True, transform=trafos
    )
    post_test_ood_data = datasets.KMNIST(
        datadir + "kmnist", train=False, download=True, transform=trafos
    )

    train_data, ood_data = prepare_for_training_with_ood(train_data, ood_data)
    test_data, test_ood_data = prepare_for_training_with_ood(test_data, test_ood_data)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )
    ood_loader = DataLoader(
        ood_data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )
    test_ood_loader = DataLoader(
        test_ood_data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )
    post_test_ood_loader = DataLoader(
        post_test_ood_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **kwargs
    )

    return train_loader, test_loader, ood_loader, test_ood_loader, post_test_ood_loader


def prepare_data_cifar10(batch_size, iscuda):
    kwargs = {"num_workers": 1, "pin_memory": True} if iscuda else {}

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    datadir = "~/Documents/data/"

    train_data = datasets.CIFAR10(
        datadir + "cifar10", train=True, download=True, transform=transform_train
    )
    test_data = datasets.CIFAR10(
        datadir + "cifar10", train=False, download=True, transform=transform_test
    )
    ood_data = datasets.CIFAR100(
        datadir + "cifar100", train=True, download=True, transform=transform_train
    )
    test_ood_data = datasets.CIFAR100(
        datadir + "cifar100", train=False, download=True, transform=transform_test
    )
    post_test_ood_data = datasets.SVHN(
        datadir + "svhn", split="test", download=True, transform=transform_test
    )

    train_data, ood_data = prepare_for_training_with_ood(train_data, ood_data)
    test_data, test_ood_data = prepare_for_training_with_ood(test_data, test_ood_data)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs
    )
    ood_loader = DataLoader(
        ood_data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )
    test_ood_loader = DataLoader(
        test_ood_data, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs
    )
    post_test_ood_loader = DataLoader(
        post_test_ood_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        **kwargs
    )

    return train_loader, test_loader, ood_loader, test_ood_loader, post_test_ood_loader


def prepare_data_svhn(batch_size, iscuda):
    kwargs = {"num_workers": 1, "pin_memory": True} if iscuda else {}

    trafos = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4377, 0.4438, 0.4728), std=(0.1980, 0.2010, 0.1970)
            ),
        ]
    )

    datadir = "~/Documents/data/"

    train_data = datasets.SVHN(
        datadir + "svhn", split="train", download=True, transform=trafos
    )
    test_data = datasets.SVHN(
        datadir + "svhn", split="test", download=True, transform=trafos
    )
    ood_data = datasets.CIFAR10(
        datadir + "cifar10", train=True, download=True, transform=trafos
    )
    test_ood_data = datasets.CIFAR10(
        datadir + "cifar10", train=False, download=True, transform=trafos
    )
    post_test_ood_data = datasets.CIFAR100(
        datadir + "cifar100", train=False, download=True, transform=trafos
    )

    train_data, ood_data = prepare_for_training_with_ood(train_data, ood_data)
    test_data, test_ood_data = prepare_for_training_with_ood(test_data, test_ood_data)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )
    ood_loader = DataLoader(
        ood_data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )
    test_ood_loader = DataLoader(
        test_ood_data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )
    post_test_ood_loader = DataLoader(
        post_test_ood_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **kwargs
    )

    return train_loader, test_loader, ood_loader, test_ood_loader, post_test_ood_loader


def prepare_data_cifar100(batch_size, iscuda):
    kwargs = {"num_workers": 1, "pin_memory": True} if iscuda else {}

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    datadir = "~/Documents/data/"

    train_data = datasets.CIFAR100(
        datadir + "cifar100", train=True, download=True, transform=transform_train
    )
    test_data = datasets.CIFAR100(
        datadir + "cifar100", train=False, download=True, transform=transform_test
    )
    ood_data = datasets.CIFAR10(
        datadir + "cifar10", train=True, download=True, transform=transform_train
    )
    test_ood_data = datasets.CIFAR10(
        datadir + "cifar10", train=False, download=True, transform=transform_test
    )
    post_test_ood_data = datasets.SVHN(
        datadir + "svhn", split="test", download=True, transform=transform_test
    )

    train_data, ood_data = prepare_for_training_with_ood(train_data, ood_data)
    test_data, test_ood_data = prepare_for_training_with_ood(test_data, test_ood_data)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )
    ood_loader = DataLoader(
        ood_data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )
    test_ood_loader = DataLoader(
        test_ood_data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )
    post_test_ood_loader = DataLoader(
        post_test_ood_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **kwargs
    )

    return train_loader, test_loader, ood_loader, test_ood_loader, post_test_ood_loader


def prepare_data(data_set, batch_size, iscuda):
    print(data_set)
    # Pick data set
    if data_set == "mnist":
        (
            train_loader,
            test_loader,
            ood_loader,
            test_ood_loader,
            post_test_ood_loader,
        ) = prepare_data_mnist(batch_size=batch_size, iscuda=iscuda)
        n_channel = 1
        n_classes = 10
    elif data_set == "fashion":
        (
            train_loader,
            test_loader,
            ood_loader,
            test_ood_loader,
            post_test_ood_loader,
        ) = prepare_data_fashion(batch_size=batch_size, iscuda=iscuda)
        n_channel = 1
        n_classes = 10
    elif data_set == "c10":
        (
            train_loader,
            test_loader,
            ood_loader,
            test_ood_loader,
            post_test_ood_loader,
        ) = prepare_data_cifar10(batch_size=batch_size, iscuda=iscuda)
        n_channel = 3
        n_classes = 10
    elif data_set == "svhn":
        (
            train_loader,
            test_loader,
            ood_loader,
            test_ood_loader,
            post_test_ood_loader,
        ) = prepare_data_svhn(batch_size=batch_size, iscuda=iscuda)
        n_channel = 3
        n_classes = 10
    elif data_set == "c100":
        (
            train_loader,
            test_loader,
            ood_loader,
            test_ood_loader,
            post_test_ood_loader,
        ) = prepare_data_cifar100(batch_size=batch_size, iscuda=iscuda)
        n_channel = 3
        n_classes = 100
    else:
        raise NotImplementedError

    return (
        train_loader,
        test_loader,
        ood_loader,
        test_ood_loader,
        post_test_ood_loader,
        n_channel,
        n_classes,
    )
