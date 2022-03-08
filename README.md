# Evidential Turing Processes

This repository includes the code to replicate the image classification experiments from the preprint:

> **Evidential Turing Processes**\
> _Melih Kandemir, Abdullah Akgül, Manuel Haussmann, Gozde Unal_\
> International Conference on Learning Representations, 2022  
> [OpenReview](https://openreview.net/forum?id=84NMXTHYe-)  
> [ArXiv](https://arxiv.org/abs/2106.01216)



## Fashion MNIST
To train the ETP run the following command

```
# for ETP with LeNet5:
python script.py --model etp --arch lenet5 --dataset fashion --max_epochs 50 --max_replication 10 --resume false
```



# Cite as
```
@inproceedings{
kandemir2022evidential,
title={Evidential Turing Processes },
author={Melih Kandemir and Abdullah Akg{\"u}l and Manuel Haussmann and Gozde Unal},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=84NMXTHYe-}
}
```
