import torch as th
import torch.nn.functional as F
from scipy.special import digamma
from torch import nn, optim
from torch.distributions import Dirichlet
from copy import deepcopy
from torch.distributions.kl import kl_divergence
import numpy as np

##############################################################################################################
# Family that models uncertainty by weight perturbation
class BayesianNeuralNet(nn.Module):
    def __init__(self, arch=None, isvb=False):
        super(BayesianNeuralNet, self).__init__()
        self.arch = deepcopy(arch)
        self.isvb = isvb

    def KL(self):
        return sum(l.KL() for l in [self.arch.parameters()] if hasattr(l, "KL"))

    def forward(self, data):
        return self.arch(data)

    def ood_predict(self, data, logits):
        _, probs = self.predict(data)
        return -th.sum(probs * th.log(probs + 1e-8), 1)

    def predict(self, data):
        out = self.forward(data)
        probs = (
            sum(self.class_probs(out) for _ in range(self.arch.n_samples))
            / self.arch.n_samples
        )
        classes = probs.max(1, keepdim=True)[1]
        return classes, probs

    def class_probs(self, alpha):
        return F.softmax(alpha, 1)

    def loss(self, data, target, x_ood, epoch):
        scores = self.forward(data)
        loss = F.cross_entropy(scores, target)
        if self.isvb:
            loss += self.KL()
        return loss


#
##############################################################################################################
class EDL(nn.Module):
    def __init__(self, arch=None):
        super(EDL, self).__init__()
        self.arch = deepcopy(arch)
        # self.isvb = isvb

    def KL(self):
        return sum(l.KL() for l in [self.arch.parameters()] if hasattr(l, "KL"))

    def forward(self, data):
        return self.arch(data).clamp(max=10)

    def ood_predict(self, data, logits):
        logits = np.asarray(logits.detach().cpu(), dtype=np.float64)
        alphas = np.exp(logits) + 1
        alpha0 = np.sum(alphas, axis=1, keepdims=True)
        probs = alphas / alpha0

        entropy_of_exp = -np.sum(probs * np.log(probs + 1e-8), axis=1)
        expected_entropy = -np.sum(
            (alphas / alpha0) * (digamma(alphas + 1) - digamma(alpha0 + 1.0)), axis=1
        )
        return th.tensor(entropy_of_exp - expected_entropy).cuda()

    def predict(self, data):
        out = self.forward(data)
        alpha = th.exp(out) + 1
        preds = self.class_probs(alpha)
        classes = preds.max(1, keepdim=True)[1]
        return classes, preds

    def class_probs(self, alpha):
        return alpha / alpha.sum(1, keepdims=True)

    def loss(self, data, target, x_ood, epoch):
        def kl_dirichlet(alphas):
            q = Dirichlet(alphas)
            p = Dirichlet(th.ones_like(alphas))
            return kl_divergence(q, p)

        anneal = min(1, (epoch - 1) / 10.0)

        scores = self.forward(data)
        n_class = scores.shape[1]
        target = F.one_hot(target, self.arch.n_classes)

        alphas = th.exp(scores) + 1
        S = alphas.sum(1, keepdims=True)
        preds = alphas / S
        fit_data = (
            ((target - preds).pow(2) + preds * (1 - preds) / (S + 1)).sum(1).mean()
        )

        alpha_tilde = target + (1 - target) * alphas
        reg = anneal * kl_dirichlet(alpha_tilde).mean()

        return fit_data + 0.1 * reg


#
##############################################################################################################
class RPN(nn.Module):
    def __init__(self, arch=None, isvb=False):
        super(RPN, self).__init__()
        self.arch = deepcopy(arch)
        self.isvb = isvb

        self.id_loss = self.DirichletKLLoss(
            target_concentration=1e2, concentration=1.0, reverse=True
        )
        self.ood_loss = self.DirichletKLLoss(
            target_concentration=0.0, concentration=1.0, reverse=True
        )

        self.criterion = self.PriorNetMixedLoss(
            [self.id_loss, self.ood_loss], mixing_params=[1.0, 1.0]
        )

    def forward(self, data):
        return self.arch(data)

    def predict(self, data):
        out = self.forward(data)
        probs = F.softmax(out, dim=1)
        classes = probs.max(1, keepdim=True)[1]
        return classes, probs

    def ood_predict(self, data, logits):
        logits = np.asarray(logits.detach().cpu(), dtype=np.float64)
        alphas = np.exp(logits)
        alpha0 = np.sum(alphas, axis=1, keepdims=True)
        probs = alphas / alpha0

        entropy_of_exp = -np.sum(probs * np.log(probs + 1e-8), axis=1)
        expected_entropy = -np.sum(
            (alphas / alpha0) * (digamma(alphas + 1) - digamma(alpha0 + 1.0)), axis=1
        )
        return th.tensor(entropy_of_exp - expected_entropy).cuda()

    def class_probs(self, alpha):
        return F.softmax(alpha, 1)

    def loss(self, data, target, x_ood, epoch):
        logits_id = self.forward(data)
        logits_ood = self.forward(x_ood)
        loss = self.criterion((logits_id, logits_ood), (target, None))
        th.nn.utils.clip_grad_norm_(self.arch.parameters(), 10.0)
        return loss

    class PriorNetMixedLoss:
        def __init__(self, losses, mixing_params=None):

            self.losses = losses
            if mixing_params is not None:
                self.mixing_params = mixing_params
            else:
                self.mixing_params = [1.0] * len(self.losses)

        def __call__(self, logits_list, labels_list):
            return self.forward(logits_list, labels_list)

        def forward(self, logits_list, labels_list):
            total_loss = []
            target_concentration = 0.0
            for i, loss in enumerate(self.losses):
                if loss.target_concentration > target_concentration:
                    target_concentration = loss.target_concentration
                weighted_loss = (
                    loss(logits_list[i], labels_list[i]) * self.mixing_params[i]
                )
                total_loss.append(weighted_loss)
            total_loss = th.stack(total_loss, dim=0)
            return th.sum(total_loss) / target_concentration

    class DirichletKLLoss:
        def __init__(self, target_concentration=1e3, concentration=1.0, reverse=True):

            self.target_concentration = th.tensor(
                target_concentration, dtype=th.float32
            )
            self.concentration = concentration
            self.reverse = reverse

        def __call__(self, logits, labels, reduction="mean"):
            alphas = th.exp(logits)
            return self.forward(alphas, labels, reduction=reduction)

        def forward(self, alphas, labels, reduction="mean"):
            loss = self.compute_loss(alphas, labels)

            if reduction == "mean":
                return th.mean(loss)
            elif reduction == "none":
                return loss
            else:
                raise NotImplementedError

        def compute_loss(self, alphas, labels=None):
            target_alphas = th.ones_like(alphas) * self.concentration
            if labels is not None:
                target_alphas += (
                    F.one_hot(labels, alphas.shape[1]) * self.target_concentration
                )

            if self.reverse:
                loss = self.dirichlet_reverse_kl_divergence(
                    alphas=alphas, target_alphas=target_alphas
                )
            else:
                loss = self.dirichlet_kl_divergence(
                    alphas=alphas, target_alphas=target_alphas
                )
            return loss

        def dirichlet_kl_divergence(
            self,
            alphas,
            target_alphas,
            precision=None,
            target_precision=None,
            epsilon=1e-8,
        ):
            if not precision:
                precision = th.sum(alphas, dim=1, keepdim=True)
            if not target_precision:
                target_precision = th.sum(target_alphas, dim=1, keepdim=True)

            precision_term = th.lgamma(target_precision) - th.lgamma(precision)
            alphas_term = th.sum(
                th.lgamma(alphas + epsilon)
                - th.lgamma(target_alphas + epsilon)
                + (target_alphas - alphas)
                * (
                    th.digamma(target_alphas + epsilon)
                    - th.digamma(target_precision + epsilon)
                ),
                dim=1,
                keepdim=True,
            )

            cost = th.squeeze(precision_term + alphas_term)
            return cost

        def dirichlet_reverse_kl_divergence(
            self,
            alphas,
            target_alphas,
            precision=None,
            target_precision=None,
            epsilon=1e-8,
        ):
            return self.dirichlet_kl_divergence(
                alphas=target_alphas,
                target_alphas=alphas,
                precision=target_precision,
                target_precision=precision,
                epsilon=epsilon,
            )


#


##############################################################################################################
# TS
# taken from "https://github.com/gpleiss/temperature_scaling"
class TS(nn.Module):
    def __init__(self, arch):
        super(TS, self).__init__()
        self.arch = deepcopy(arch)
        self.temperature = nn.Parameter(th.ones(1) * 1.5)
        self.scaled = False

    def forward(self, data):
        return self.arch(data)

    def ood_predict(self, data, logits):
        probs = F.softmax(logits)
        return -th.sum(probs * th.log(probs + 1e-8), 1)

    def predict(self, data):
        if self.scaled:
            out = self._forward(data)
        else:
            out = self.forward(data)
        probs = (
            sum(self.class_probs(out) for _ in range(self.arch.n_samples))
            / self.arch.n_samples
        )
        classes = probs.max(1, keepdim=True)[1]
        return classes, probs

    def class_probs(self, alpha):
        return F.softmax(alpha, 1)

    def loss(self, data, target, x_ood, epoch):
        scores = self.forward(data)
        loss = F.cross_entropy(scores, target)
        return loss

    def _forward(self, input):
        logits = self.arch(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(
            logits.size(0), logits.size(1)
        )
        return logits / temperature

    def set_temperature(self, valid_loader):
        self.scaled = True
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()

        logits_list = []
        labels_list = []
        with th.no_grad():
            for input, label in valid_loader:
                input = input.cuda()
                logits = self._forward(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = th.cat(logits_list).cuda()
            labels = th.cat(labels_list).cuda()

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)
        return self


##############################################################################################################
