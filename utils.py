from scores import *


def get_results(
    model,
    test_loader,
    ood_loader,
    test_ood_loader,
    post_test_ood_loader,
    iscuda,
    batch_size=200,
):
    _err = compute_score(model, test_loader, err, "Error", "test", iscuda).item()
    _nll = compute_score(model, test_loader, nll, "NLL", "test", iscuda).item()
    _ece = compute_score(model, test_loader, ece, "ECE", "test", iscuda)
    _auroc = compute_score_ood(
        model,
        test_loader,
        post_test_ood_loader,
        auroc,
        "AU-ROC",
        "ood post test",
        iscuda,
    )

    return {
        "err": _err,
        "nll": _nll,
        "ece": _ece,
        "auroc": _auroc,
    }


