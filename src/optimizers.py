import torch
from torch.optim import SGD
from torch import Tensor
from typing import List, Optional
from math import prod
import torch.nn.functional as F


def gram_schmidt_tensor(X):
    assert X.ndim == 4
    shape = X.shape
    Q, _ = torch.linalg.qr(X.view(shape[0], -1).T)

    if shape[0] <= prod(shape[1:]):
        return Q.T.view(shape)
    else:
        return (F.pad(Q, (0, X.shape[0]-Q.shape[1])).T).view(shape)


def sgd_l2_proj(params: List[Tensor],
                d_p_list: List[Tensor],
                momentum_buffer_list: List[Optional[Tensor]],
                *,
                weight_decay: float,
                momentum: float,
                lr: float,
                dampening: float,
                nesterov: bool,
                GS: bool):

    for i, param in enumerate(params):

        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        if param.ndim == 4:
            norms_before = torch.norm(param, p=2, dim=tuple(
                range(1, param.ndim)), keepdim=True)

            # project it on the tangent plane of l2 ball
            d_p -= (torch.sum(d_p * param, dim=tuple(range(1, param.ndim)))/torch.sum(
                param**2, dim=tuple(range(1, param.ndim)))).view((-1, *[1]*(param.ndim-1)))*param

        param.add_(d_p, alpha=-lr)

        if param.ndim == 4:
            # normalize the param
            if GS:
                param = gram_schmidt_tensor(param)
            else:
                param.divide_(torch.norm(param, p=2, dim=tuple(
                    range(1, param.ndim)), keepdim=True))

            param.multiply_(norms_before)


class SGD_l2_proj(SGD):

    def __init__(self, params, GS=False, **kwargs):
        super(SGD_l2_proj, self).__init__(params, **kwargs)
        self.defaults['GS'] = GS
        for param_group in self.param_groups:
            param_group.setdefault('GS', GS)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']
            GS = group['GS']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            sgd_l2_proj(params_with_grad,
                        d_p_list,
                        momentum_buffer_list,
                        weight_decay=weight_decay,
                        momentum=momentum,
                        lr=lr,
                        dampening=dampening,
                        nesterov=nesterov,
                        GS=GS)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss
