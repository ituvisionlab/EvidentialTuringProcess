import os
import numpy as np
import torch as th
from dataprep import prepare_data
from architectures import LeNet5
from etp import EvidentialTuringProcess
import argparse
import time
from datetime import timedelta
import json

from utils import get_results


def train(model, train_loader, ood_loader, epoch, opt, iscuda=False, verbose=False):
    model.train()

    total_loss = 0.0
    for (data, target), (ood_data, ood_target) in zip(train_loader, ood_loader):
        if iscuda:
            data, target = data.cuda(), target.cuda()
            ood_data, ood_target = ood_data.cuda(), ood_target.cuda()
        opt.zero_grad()

        loss = model.loss(data, target)

        loss.backward()
        total_loss += loss.item()
        opt.step()

    if verbose:
        print(f"{epoch}: AvgLoss = {total_loss / len(train_loader):.010f}")

    return opt


def test_acc(model, loader, post_test_ood_loader, epoch, label="Test", iscuda=False):
    model.eval()
    correct = 0
    with th.no_grad():
        for data, target in loader:
            if iscuda:
                data, target = data.cuda(), target.cuda()
            pred_cls, pred_probs = model.predict(data)
            correct += pred_cls.eq(target.view_as(pred_cls)).sum().item()
    print(
        f"{epoch}: {label} set: "
        f"Accuracy: {correct}/{len(loader.dataset)} ({100. * correct / len(loader.dataset):.02f}%) "
        f"Error(%) = {100 * (1 - correct / len(loader.dataset)):.02f}"
    )

    if np.mod(epoch, 5) == 0:
        get_results(model, loader, None, None, post_test_ood_loader, True, 128)


def main(
    model_name="mcdrop",
    arch_name="lenet5",
    max_epochs=20,
    data_set="mnist",
    exp_num=0,
):
    batch_size = 128
    (
        train_loader,
        test_loader,
        ood_loader,
        test_ood_loader,
        post_test_ood_loader,
        n_channel,
        n_classes,
    ) = prepare_data(data_set, batch_size, iscuda)
    result = {"train": {"epoch_time": 0}, "test": {}}

    # Pick architecture
    arch = LeNet5(
        n_channel,
        n_classes,
        isvb=True,
        has_context=False,
    )

    n_data = len(train_loader.dataset)
    arch.dataset_size = n_data

    # Pick model
    model = EvidentialTuringProcess(arch)

    lrate = 1e-3
    opt = th.optim.Adam(model.parameters(), lr=lrate)

    # Pick device
    if iscuda:
        model.cuda()

    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

    # Training routine
    for epoch in range(1, max_epochs + 1):
        start = time.time()
        opt = train(
            model, train_loader, ood_loader, epoch, opt, iscuda=iscuda, verbose=True
        )
        result["train"]["epoch_time"] += time.time() - start
        test_acc(model, test_loader, post_test_ood_loader, epoch, iscuda=iscuda)
        scheduler.step()

    result["test"] = get_results(
        model,
        test_loader,
        ood_loader,
        test_ood_loader,
        post_test_ood_loader,
        iscuda,
        batch_size,
    )
    result["train"]["epoch_time"] = timedelta(
        seconds=result["train"]["epoch_time"] / max_epochs
    )

    return result, model, opt


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="etp"
    )
    parser.add_argument("--arch", default="lenet5")
    parser.add_argument(
        "--dataset",
        default="fashion"
    )
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--max_replication", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    print(args)
    arch = args.arch
    model_name = args.model
    dataset = args.dataset
    max_epochs = args.max_epochs
    max_replication = args.max_replication

    os.makedirs("runs", exist_ok=True)
    os.makedirs("runs/baselines", exist_ok=True)
    os.makedirs(f"runs/baselines/{arch}", exist_ok=True)
    os.makedirs(f"runs/baselines/{arch}/{dataset}", exist_ok=True)
    os.makedirs(f"runs/baselines/{arch}/{dataset}/{model_name}", exist_ok=True)

    if th.cuda.is_available():
        th.backends.cudnn.benchmark = True
        iscuda = True
    else:
        iscuda = False

    avg_err = []
    avg_nll = []
    avg_ece = []
    avg_auroc = []

    for exp_num in range(max_replication):
        try:
            start = time.time()
            result, model, optimizer = main(
                arch_name=arch,
                model_name=model_name,
                max_epochs=max_epochs,
                data_set=dataset,
                exp_num=exp_num,
            )

            result["train"]["total_time"] = timedelta(seconds=time.time() - start)

            avg_err.append(result["test"]["err"])
            avg_nll.append(result["test"]["nll"])
            avg_ece.append(result["test"]["ece"])
            avg_auroc.append(result["test"]["auroc"])

            replication_name = f"replication_{exp_num + 1}"
            state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            os.makedirs(
                f"runs/baselines/{arch}/{dataset}/{model_name}/{replication_name}/",
                exist_ok=True,
            )
            th.save(
                state,
                f"runs/baselines/{arch}/{dataset}/{model_name}/{replication_name}/checkpoint.pt",
            )

            with open(
                f"runs/baselines/{arch}/{dataset}/{model_name}/{replication_name}/train.txt",
                "w",
            ) as f:
                json.dump(result["train"], f, default=str)

            with open(
                f"runs/baselines/{arch}/{dataset}/{model_name}/{replication_name}/test.txt",
                "w",
            ) as f:
                json.dump(result["test"], f, default=str)

        except Exception as e:
            print("error\n", e)

    with open(f"runs/baselines/{arch}/{dataset}/{model_name}/stats.txt", "w") as f:
        avg_err = np.array(avg_err)
        avg_nll = np.array(avg_nll)
        avg_ece = np.array(avg_ece)
        avg_auroc = np.array(avg_auroc)

        printer = lambda lst: f"Mean: {lst.mean()}\t Std:{lst.std()}"
        to_print = f"Err:\t{printer(avg_err)}\n"
        to_print += f"Nll:\t{printer(avg_nll)}\n"
        to_print += f"Ece:\t{printer(avg_ece)}\n"
        to_print += f"Auroc:\t{printer(avg_auroc)}\n"

        f.write(to_print)
