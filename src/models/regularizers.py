import torch


def hoyer_regularizer(x):
    return torch.norm(x, p=1, dim=tuple(range(x.ndim)))/torch.norm(x, p=2, dim=tuple(range(x.ndim)))


def hoyer_square_regularizer(x):
    return hoyer_regularizer(x)**2


def hoyer_per_img_regularizer(x):
    return (torch.norm(x, p=1, dim=tuple(range(1, x.ndim)))/torch.norm(x, p=2, dim=tuple(range(1, x.ndim)))).sum()


def hoyer_per_img_square_regularizer(x):
    return ((torch.norm(x, p=1, dim=tuple(range(1, x.ndim)))/torch.norm(x, p=2, dim=tuple(range(1, x.ndim))))**2).sum()
