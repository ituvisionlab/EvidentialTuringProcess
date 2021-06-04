# Evidential Turing Processes

This repository includes the code to replicate the image classification experiments from the preprint:

> **Evidential Turing Processes**\
> _Melih Kandemir, Abdullah Akgül, Manuel Haussmann, Gozde Unal_\
> [ArXiv](https://arxiv.org/abs/2106.01216)



## Fashion MNIST
To train the individual models run the following commands

```
# for ETP with LeNet5:
python script.py --model etp --arch lenet5 --dataset fashion --max_epochs 50 --max_replication 10 --resume false
# for MCDrop with LeNet5:
python script.py --model mcdrop --arch lenet5 --dataset fashion --max_epochs 50 --max_replication 10 --resume false
# for BNN-VB with LeNet5:
python script.py --model vb --arch lenet5 --dataset fashion --max_epochs 50 --max_replication 10 --resume false
# for EDL with LeNet5:
python script.py --model edl --arch lenet5 --dataset fashion --max_epochs 50 --max_replication 10 --resume false
# for TS with LeNet5:
python script.py --model ts --arch lenet5 --dataset fashion --max_epochs 50 --max_replication 10 --resume false
# for PN-RKL with LeNet5:
python script.py --model rpn --arch lenet5 --dataset fashion --max_epochs 50 --max_replication 10 --resume false
```

## CIFAR 10
To train the individual models run the following commands

```
# for ETP with LeNet5:
python script.py --model etp --arch lenet5 --dataset c10 --max_epochs 100 --max_replication 10 --resume false
# for MCDrop with LeNet5:
python script.py --model mcdrop --arch lenet5 --dataset c10 --max_epochs 100 --max_replication 10 --resume false
# for BNN-VB with LeNet5:
python script.py --model vb --arch lenet5 --dataset c10 --max_epochs 100 --max_replication 10 --resume false
# for EDL with LeNet5:
python script.py --model edl --arch lenet5 --dataset c10 --max_epochs 100 --max_replication 10 --resume false
# for TS with LeNet5:
python script.py --model ts --arch lenet5 --dataset c10 --max_epochs 100 --max_replication 10 --resume false
# for PN-RKL with LeNet5:
python script.py --model rpn --arch lenet5 --dataset c10 --max_epochs 100 --max_replication 10 --resume false
```

## CIFAR 10
To train the individual models run the following commands

```
# for ETP with LeNet5:
python script.py --model etp --arch resnet18 --dataset c100 --max_epochs 200 --max_replication 10 --resume false
# for MCDrop with LeNet5:
python script.py --model mcdrop --arch resnet18 --dataset c100 --max_epochs 200 --max_replication 10 --resume false
# for BNN-VB with LeNet5:
python script.py --model vb --arch resnet18 --dataset c100 --max_epochs 200 --max_replication 10 --resume false
# for EDL with LeNet5:
python script.py --model edl --arch resnet18 --dataset c100 --max_epochs 200 --max_replication 10 --resume false
# for TS with LeNet5:
python script.py --model ts --arch resnet18 --dataset c100 --max_epochs 200 --max_replication 10 --resume false
# for PN-RKL with LeNet5:
python script.py --model rpn --arch resnet18 --dataset c100 --max_epochs 200 --max_replication 10 --resume false
```
