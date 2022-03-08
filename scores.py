import numpy as np
import torch as th
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def nll(preds, target, minibatch=True):
    logpred = th.log(preds + 1e-8)
    if minibatch:
        return -(logpred * target).sum(1)
    else:
        return -(logpred * target).sum(1).mean()


def err(preds, target, minibatch=True):
    preds = preds.argmax(1)
    target = target.argmax(1)
    if minibatch:
        return ((preds != target) * 1.0).sum() * 100
    else:
        return ((preds != target) * 1.0).mean() * 100


def auroc(preds, target, minibatch=True):
    score = roc_auc_score(target, preds)
    return score


def entropy(preds, target, minibatch=True):
    if minibatch:
        return -(preds * th.log(preds + 1e-8)).sum()
    else:
        return -(preds * th.log(preds + 1e-8)).mean()


def ece(preds, target, minibatch=True):
    confidences, predictions = th.max(preds, 1)
    _, target_cls = th.max(target, 1)
    accuracies = predictions.eq(target_cls)
    n_bins = 100  # 30000
    bin_boundaries = th.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = th.zeros(1, device="cuda")
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += th.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin * 100

    return ece.item()


def compute_score_ood(
    model, loader_in, loader_out, score, name_score="foo", name_data="bar", iscuda=False
):
    preds_in = []
    preds_ood = []
    with th.no_grad():
        for data, target in loader_in:
            if iscuda:
                data = data.cuda()
            try:
                # for ts
                pred_prob = model._forward(data)
            except:
                pred_prob = model.forward(data)
            ood_pred_prob = model.ood_predict(data, pred_prob)
            preds_in.append(ood_pred_prob.cpu().numpy())

    with th.no_grad():
        for data, target in loader_out:
            if iscuda:
                data = data.cuda()
            try:
                # for ts
                pred_prob = model._forward(data)
            except:
                pred_prob = model.forward(data)
            ood_pred_prob = model.ood_predict(data, pred_prob)
            preds_ood.append(ood_pred_prob.cpu().numpy())

    preds_in = np.concatenate(preds_in, axis=0)
    preds_ood = np.concatenate(preds_ood, axis=0)
    logits = np.concatenate([preds_in, preds_ood], axis=0)

    in_domain = np.zeros(shape=[preds_in.shape[0]], dtype=np.int32)
    ood_domain = np.ones(shape=[preds_ood.shape[0]], dtype=np.int32)
    domain_labels = np.concatenate([in_domain, ood_domain], axis=0)

    result = score(logits, domain_labels, minibatch=False)

    print(f"The {name_score} score on the {name_data} set is: {result:.04f}")
    return result


def compute_score(
    model, loader, score, name_score="foo", name_data="bar", iscuda=False
):
    preds = []
    targets = []
    with th.no_grad():
        for data, target in loader:
            if iscuda:
                data, target = data.cuda(), target.cuda()

            target = F.one_hot(target, model.arch.n_classes)
            _, pred_prob = model.predict(data)
            preds.append(pred_prob)
            targets.append(target)

    targets = th.cat(targets).cuda()
    preds = th.cat(preds).cuda()
    result = score(preds, targets, minibatch=False)

    print(f"The {name_score} score on the {name_data} set is: {result:.04f}")
    return result
