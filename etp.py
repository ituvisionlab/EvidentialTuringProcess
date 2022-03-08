import torch as th
import torch.nn.functional as F
from scipy.special import digamma
from torch import nn
from torch.distributions import Dirichlet, Normal, Gamma
from torch.distributions.kl import kl_divergence
from copy import deepcopy
import math
import numpy as np


class ETPHyperParams:
    def __init__(self, n_classes=None):
        super(ETPHyperParams, self).__init__()
        self.memory_learning_rate = 0.99
        self.memory_size = 20
        self.anneal_factor = 1e-3
        self.memo_variance = 0.1


class EvidentialTuringProcess(nn.Module):
    def __init__(self, arch=None):
        super(EvidentialTuringProcess, self).__init__()

        self.arch = deepcopy(arch)
        self.hyperparams = ETPHyperParams(self.arch.n_classes)

        x_dim = [28, 32][self.arch.n_channels == 3];
        y_dim = x_dim
        self.x_dim = x_dim;
        self.y_dim = y_dim;
        self.n_channels = self.arch.n_channels

        self.memory = nn.Parameter(th.Tensor(self.hyperparams.memory_size, self.arch.n_classes),
                                   requires_grad=False).cuda()
        self.memory.data.normal_(0, 0.01)
        self.memory.data.pow_(2)

        self.fc1_enc_to_pred = nn.Linear(self.arch.n_classes * 2, self.arch.n_classes)
        self.fc1_key = nn.Linear(self.arch.n_classes, self.arch.n_classes)

    def KL(self):
        return sum(l.KL() for l in [self.arch.parameters()] if hasattr(l, "KL"))

    def update_memory(self, x_embed, y, max_size=50):
        n_context = np.random.randint(3, max_size)
        x_given_embed = x_embed[:n_context, :]
        y_given = y[:n_context].view(-1, 1)

        mem_sample = self.get_memory_sample()

        new_element = F.one_hot(y_given, self.arch.n_classes).view(-1, self.arch.n_classes) + th.softmax(x_given_embed,
                                                                                                         1)
        weight_new_element = self.get_attention_weights(x_given_embed, mem_sample)
        add_new_element = th.mm(weight_new_element.transpose(0, 1), new_element)
        gamma = self.hyperparams.memory_learning_rate

        mem_offset = self.memory * (gamma - 1) + add_new_element * (1 - gamma)

        self.memory.data.add_(mem_offset)
        self.memory.data.tanh_()

    def get_memory_sample(self):
        sig2 = self.hyperparams.memo_variance
        sig2_vec = th.ones(self.memory.shape).cuda() * sig2
        return Normal(self.memory, sig2_vec).rsample()

    def get_attention_weights(self, x_embed, mem_sample):
        keys = self.fc1_key(mem_sample)
        kq = th.mm(x_embed, keys.transpose(0, 1)) / np.sqrt(self.arch.n_classes)
        return F.softmax(kq, 1)

    def get_attention(self, x_embed):
        mem_sample = self.get_memory_sample()
        weights = self.get_attention_weights(x_embed, mem_sample)
        return th.mm(weights, mem_sample)

    def ood_predict(self, data, logits):
        probs = sum(F.softmax(logits, 1) for _ in range(self.arch.n_samples)) / self.arch.n_samples
        entropy_of_exp = -th.sum(probs * th.log(probs + 1e-8), axis=1)
        alpha = th.exp(logits)
        S = alpha.sum(1, keepdims=True)
        expected_entropy = -th.sum((alpha / S) * (th.digamma(alpha + 1) - th.digamma(S + 1.0)), axis=1)
        return entropy_of_exp  # - expected_entropy

    def predict(self, data):
        x_embed = self.forward(data)
        alpha = th.exp(x_embed)
        S = alpha.sum(1, keepdims=True)
        probs = alpha / S
        classes = probs.max(1, keepdim=True)[1]
        return classes, probs

    def forward(self, input):
        x_embed = self.arch(input)
        attention = self.get_attention(x_embed)
        logit = self.fc1_enc_to_pred(th.cat((x_embed, attention), dim=1))
        return logit.clamp(max=15)

    def kl_dirichlet(self, alpha, beta):
        q = Dirichlet(alpha)
        p = Dirichlet(beta)
        return kl_divergence(q, p)

    def loss(self, x, y):
        y_one_hot = F.one_hot(y, self.arch.n_classes).view(-1, self.arch.n_classes)
        x_embed_pre = self.arch(x)
        attention = self.get_attention(x_embed_pre)
        logit = self.fc1_enc_to_pred(th.cat((x_embed_pre, attention), dim=1))
        x_embed = logit.clamp(max=15)

        self.update_memory(x_embed_pre, y)

        alpha = th.exp(x_embed)
        S = alpha.sum(1, keepdims=True)
        fit_term = (y_one_hot * (th.digamma(S + 1e-8) - th.digamma(alpha + 1e-8))).sum(axis=1)
        reg_term = self.kl_dirichlet(alpha, th.exp(attention))
        loss = (fit_term + reg_term * self.hyperparams.anneal_factor).mean()
        loss += self.KL() / self.arch.dataset_size

        return loss