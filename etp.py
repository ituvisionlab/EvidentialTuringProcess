import torch as th
import torch.nn.functional as F
from torch import nn
from torch.distributions import Dirichlet, Normal
from torch.distributions.kl import kl_divergence
from copy import deepcopy
import numpy as np


class ETPHyperParams:
    def __init__(self, n_classes=None):
        super(ETPHyperParams, self).__init__()

        if n_classes < 20:
            self.anneal_factor = 1e-3
            self.is_drop = False
        else:
            self.anneal_factor = (
                1e-3
            )
            self.is_drop = True

        self.learn_beta = False
        self.memory_size = 20
        self.memory_learning_rate = 0.99
        self.decoder_width = 128
        self.eps = 1e-4


class EvidentialTuringProcess(nn.Module):
    def __init__(self, arch=None):
        super(EvidentialTuringProcess, self).__init__()

        self.arch = deepcopy(arch)

        self.hyperparams = ETPHyperParams(self.arch.n_classes)

        self.arch.drop = self.hyperparams.is_drop

        x_dim = [28, 32][self.arch.n_channels == 3]
        y_dim = x_dim

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_channels = self.arch.n_channels

        self.memory = nn.Parameter(
            th.Tensor(self.hyperparams.memory_size, self.arch.n_classes),
            requires_grad=False,
        ).cuda()
        self.memory.data.normal_(0, 0.01)
        self.memory.data.pow_(2)

        self.fc1_enc_to_pred = nn.Linear(self.arch.n_classes * 2, self.arch.n_classes)
        self.fc1_key = nn.Linear(self.arch.n_classes, self.arch.n_classes)

        self.fc3 = nn.Linear(self.arch.n_classes, self.hyperparams.decoder_width)
        self.bn2 = nn.BatchNorm1d(self.arch.n_classes)
        self.fc4_mu = nn.Linear(self.hyperparams.decoder_width, x_dim * y_dim)
        self.fc4_log_var = nn.Linear(self.hyperparams.decoder_width, 1)

    def update_memory(self, x, y, max_size=50):

        n_context = np.random.randint(3, max_size)
        x_given = x[:n_context, :]
        y_given = y[:n_context].view(-1, 1)
        x_pred = x[n_context:, :]
        y_pred = y[n_context:]

        mem_sample = self.get_memory_sample()

        # add info from the new batch
        x_given_embed = self.arch(x_given)
        w_new_elements = self.get_attention_weights(x_given_embed, mem_sample)
        y_omg = F.one_hot(y_given, self.arch.n_classes).view(
            -1, self.arch.n_classes
        ) + th.softmax(x_given_embed, 1)
        add_element = th.mm(w_new_elements.transpose(0, 1), y_omg)
        gamma = self.hyperparams.memory_learning_rate

        mem_offset = self.memory * (gamma - 1) + add_element * (1 - gamma)
        self.memory.data.add_(mem_offset)
        self.memory.data.clamp_(max=1e5)

    def get_memory_sample(self):
        sig2 = 1
        sig2_vec = th.ones(self.memory.shape).cuda() * sig2
        c_sample = Normal(self.memory, sig2_vec).rsample()
        return c_sample

    def get_attention_weights(self, x_embed, mem_sample):
        keys = self.fc1_key(mem_sample)
        kq = th.mm(x_embed, keys.transpose(0, 1)) / np.sqrt(self.arch.n_classes)
        weights = F.softmax(kq, 1)
        return weights

    def get_attention(self, x_embed):
        mem_sample = self.get_memory_sample()
        weights = self.get_attention_weights(x_embed, mem_sample)
        a = th.mm(weights, mem_sample)
        return a

    def get_self_attention(self):
        return self.get_attention(self.memory)

    def decoder(self, z):
        h = F.softplus(self.fc3(self.bn2(z)))
        h_mu = self.fc4_mu(h)
        h_log_var = self.fc4_log_var(h).clamp(min=-30, max=30)
        return h_mu, h_log_var

    def ood_predict(self, data, logits):
        probs = (
            sum(F.softmax(logits, 1) for _ in range(self.arch.n_samples))
            / self.arch.n_samples
        )
        entropy_of_exp = -th.sum(probs * th.log(probs + 1e-8), axis=1)
        return entropy_of_exp

    def predict(self, data):
        x_embed = self.forward(data)
        probs = (
            sum(F.softmax(x_embed, 1) for _ in range(self.arch.n_samples))
            / self.arch.n_samples
        )
        classes = probs.max(1, keepdim=True)[1]
        return classes, probs

    def forward(self, input):
        x_embed = self.arch(input)
        attention = self.get_attention(x_embed)
        logit = self.fc1_enc_to_pred(th.cat((x_embed, attention), dim=1))
        return logit.clamp(max=15)

    def get_id_loss(self, pi_sample, x_obs):
        x_sample_mu, x_sample_log_var = self.decoder(pi_sample)
        x_obs = x_obs.mean(1, keepdim=True).view(-1, self.x_dim * self.y_dim)
        x_sample_var = (
            th.exp(F.softplus(x_sample_log_var)) + 1
        )
        loss = (
            th.log(x_sample_var)
            + F.mse_loss(x_sample_mu, x_obs, reduction="none") / x_sample_var
        )
        return loss.mean(1)

    def loss(self, x, y, x_ood, epoch):
        def kl_dirichlet(alpha, beta):
            q = Dirichlet(alpha)
            p = Dirichlet(beta)
            return kl_divergence(q, p)

        ##################################################

        y_one_hot = F.one_hot(y, self.arch.n_classes).view(-1, self.arch.n_classes)

        self.update_memory(x, y)

        x_embed = self.forward(x)
        alpha = th.exp(x_embed)

        pi_sample = F.softmax(x_embed, 1)

        id_nll = self.get_id_loss(pi_sample, x).mean()

        id_fit = F.cross_entropy(x_embed, y)

        memo_evidence = F.softmax(self.get_attention(x_embed), 1) + 1
        ood_reg = kl_dirichlet(alpha, memo_evidence).mean()

        loss = (
            id_fit + (id_nll + ood_reg) * self.hyperparams.anneal_factor
        )

        return loss
