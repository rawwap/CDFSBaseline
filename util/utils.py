r""" Helper functions """
import random

import torch
import numpy as np


def fix_randseed(seed):
    r""" Set random seeds for reproducibility """
    if seed is None:
        seed = int(random.random() * 1e5)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mean(x):
    return sum(x) / len(x) if len(x) > 0 else 0.0


def to_cuda(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()
    return batch


def to_cpu(tensor):
    return tensor.detach().clone().cpu()

def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

class mIOU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
        self.intersection = 0
        self.union = 0

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes) 
        return hist

    def add_batch(self, predictions, gts):
        if self.num_classes > 1:
            for lp, lt in zip(predictions, gts):
                self.hist += self._fast_hist(lp.flatten(), lt.flatten())
        else:
            for lp, lt in zip(predictions, gts):
                self.intersection += np.sum(np.logical_and(lp.flatten(), lt.flatten()))
                self.union += np.sum(np.logical_or(lp.flatten(), lt.flatten()))


    def evaluate(self):
        if self.num_classes > 1:
            iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
            return np.nanmean(iu)
        else:
            return self.intersection / self.union if self.union > 0 else 0
