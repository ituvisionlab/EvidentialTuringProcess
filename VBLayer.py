import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
import math


class VBLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_prec=10, map=True):
        super(VBLinear, self).__init__()
        self.n_in = in_features
        self.n_out = out_features

        self.prior_prec = prior_prec
        self.map = map

        self.bias = nn.Parameter(th.Tensor(out_features))
        self.mu_w = Parameter(th.Tensor(out_features, in_features))
        self.logsig2_w = nn.Parameter(th.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.mu_w.size(1))
        self.mu_w.data.normal_(0, stdv)
        self.logsig2_w.data.zero_().normal_(-9, 0.001)  # var init via Louizos
        self.bias.data.zero_()

    def KL(self, loguniform=False):
        if loguniform:
            k1 = 0.63576
            k2 = 1.87320
            k3 = 1.48695
            log_alpha = self.logsig2_w - 2 * th.log(self.mu_w.abs() + 1e-8)
            kl = -th.sum(
                k1 * th.sigmoid(k2 + k3 * log_alpha) - 0.5 * F.softplus(-log_alpha) - k1
            )
        else:
            logsig2_w = self.logsig2_w.clamp(-11, 11)
            kl = (
                0.5
                * (
                    self.prior_prec * (self.mu_w.pow(2) + logsig2_w.exp())
                    - logsig2_w
                    - 1
                    - np.log(self.prior_prec)
                ).sum()
            )
        return kl

    def forward(self, input):
        # Sampling free forward pass only if MAP prediction and no training rounds
        if self.map and not self.training:
            return F.linear(input, self.mu_w, self.bias)
        else:
            mu_out = F.linear(input, self.mu_w, self.bias)
            logsig2_w = self.logsig2_w.clamp(-11, 11)
            s2_w = logsig2_w.exp()
            var_out = F.linear(input.pow(2), s2_w) + 1e-8
            return mu_out + var_out.sqrt() * th.randn_like(mu_out)

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.n_in)
            + " -> "
            + str(self.n_out)
            + ")"
        )

