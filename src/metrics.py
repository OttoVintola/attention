

import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from tqdm import tqdm 

@torch.no_grad()
def robustness_vs_noise(vit_model, dataloader, device, bayes, noise_levels=[0., 0.1, 0.25, 0.4]):
    """
    Add Gaussian noise to inputs and evaluate accuracy drop.
    Returns dict: {sigma: accuracy}
    """
    vit_model.eval()
    results = {}
    for sigma in noise_levels:
        total = 0
        correct = 0
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            noisy = images + torch.randn_like(images) * sigma
            # MC predictive mean
            T = 3
            logits_mc = 0
            for t in range(T):
                if bayes:
                    logits_mc = logits_mc + vit_model(noisy, sample=True)
                else:
                    logits_mc = logits_mc + vit_model(noisy)
            logits_mc = logits_mc / float(T)
            preds = logits_mc.argmax(dim=-1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
        results[sigma] = correct / total
    return results



def expected_calibration_error(probs, labels, n_bins=15):
    """
    Simple ECE implementation.
    probs: tensor (N, C)
    labels: tensor (N,)
    """
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)
    bins = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lb = bins[i]
        ub = bins[i + 1]
        mask = (confidences > lb) & (confidences <= ub)
        if mask.sum() == 0:
            continue
        acc = accuracies[mask].float().mean()
        conf = confidences[mask].mean()
        ece += (mask.float().mean()) * torch.abs(acc - conf)
    return ece.item()

@torch.no_grad()
def evaluate(vit_model, dataloader, device):
    vit_model.eval()
    total = 0
    correct = 0
    total_nll = 0.0
    all_probs = []
    all_labels = []

    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        # For predictive mean, average multiple samples for MC estimate
        T = 5
        logits_mc = 0
        for t in range(T):
            logits_mc = logits_mc + vit_model(images, sample=True)
        logits_mc = logits_mc / float(T)

        probs = torch.softmax(logits_mc, dim=-1)
        preds = probs.argmax(dim=-1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        total_nll += F.nll_loss(torch.log(probs + 1e-12), labels, reduction='sum').item()

        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    accuracy = correct / total
    avg_nll = total_nll / total

    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    ece_val = expected_calibration_error(all_probs, all_labels, n_bins=15)

    return {'accuracy': accuracy, 'nll': avg_nll, 'ece': ece_val, 'total': total}