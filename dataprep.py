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


def prepare_data(data_set, batch_size, iscuda):
    print(data_set)
    # Pick data set
    if data_set == "fashion":
        (
            train_loader,
            test_loader,
            ood_loader,
            test_ood_loader,
            post_test_ood_loader,
        ) = prepare_data_fashion(batch_size=batch_size, iscuda=iscuda)
        n_channel = 1
        n_classes = 10
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
