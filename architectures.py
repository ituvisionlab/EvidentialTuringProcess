import torch as th
import torch.nn.functional as F
from torch import nn
from torchvision import models

from VBLayer import VBLinear


class LeNet5(nn.Module):
    def __init__(
        self,
        n_channels=1,
        n_classes=10,
        mcdrop=True,
        isvb=False,
        has_context=False,
        prior_precision=10,
    ):
        """
        Basic LeNet5 architecture for now following the details from Louizos
        :param n_channels: 1 creates MNIST arch, 3 creates Cifar arch
        :param n_classes: 10 target classes
        :param drop: true/false switches between deterministic and dropout based version
        """
        super(LeNet5, self).__init__()

        print(isvb)

        self.drop = mcdrop
        self.isvb = isvb
        self.drop_rate = 0.25
        if mcdrop:
            self.n_samples = 10
        else:
            self.n_samples = 1
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.has_context = has_context

        if n_channels == 1:
            self.conv1 = nn.Conv2d(1, 20, 5, bias=True)
            self.conv2 = nn.Conv2d(20, 50, 5, bias=True)
            dim_cf = 4 * 4 * 50
            self.fc1 = nn.Linear(dim_cf, 500, bias=True)
            self.fc2 = nn.Linear(500, n_classes)
        elif n_channels == 3:
            self.conv1 = nn.Conv2d(3, 192, 5, bias=True)
            self.conv2 = nn.Conv2d(192, 192, 5, bias=True)

            dim_cf = 5 * 5 * 192

            if isvb:
                self.fc1 = VBLinear(dim_cf, 1000, prior_prec=prior_precision)
                self.fc2 = VBLinear(1000, n_classes, prior_prec=prior_precision)
            else:
                self.fc1 = nn.Linear(dim_cf, 1000, bias=True)
                self.fc2 = nn.Linear(1000, n_classes)
        else:
            raise NotImplementedError(f"Sorry {n_channels} is currently not possible")

    def forward(self, input, context=None):
        out = F.relu(self.conv1(input))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = th.flatten(out, 1)

        if self.has_context and context is not None:
            context_mat = th.ones(out.shape[0], self.n_classes).cuda() * context
            out = th.cat((out, context_mat), 1)

        if self.drop and not self.isvb:
            out = F.dropout(out, self.drop_rate)
        out = F.relu(self.fc1(out))
        if self.drop and not self.isvb:
            out = F.dropout(out, self.drop_rate)
        out = self.fc2(out)
        return out


class Resnet18(nn.Module):
    def __init__(
        self,
        n_channels=1,
        n_classes=10,
        mcdrop=True,
        isvb=False,
        prior_precision=10,
        has_context=False,
    ):

        super(Resnet18, self).__init__()

        print(isvb)

        self.drop = mcdrop
        self.isvb = isvb
        self.drop_rate = 0.25
        self.n_samples = 10
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.has_context = has_context

        self.model = models.resnet18()

        if n_channels == 1:
            self.model.conv1 = nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )

        in_features = self.model.fc.in_features

        if isvb:
            self.model.fc = VBLinear(in_features, n_classes, prior_prec=prior_precision)
        else:
            self.model.fc = nn.Linear(in_features, n_classes)

    def forward(self, input, context=None):
        x = self.model.relu(self.model.bn1(self.model.conv1(input)))
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        out = self.model.avgpool(self.model.layer4(x))

        out = th.flatten(out, 1)

        if self.drop and not self.isvb:
            out = F.dropout(out, self.drop_rate)

        out = self.model.fc(out)
        return out
