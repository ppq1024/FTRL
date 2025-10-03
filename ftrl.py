import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT

from typing import Union

__all__ = ["FTRL", "FTRLAdam",]


class FTRL(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        alpha: float = 1.0,
        weight_shrink: float = 0, # L1 param
        weight_decay: float = 0, # L2 param
        *,
        maximize: bool = False,
    ) -> None:
        # TODO: 参数检查

        defaults = dict(
            lr=lr,
            alpha=alpha,
            weight_shrink=weight_shrink,
            weight_decay=weight_decay,
            maximize=maximize,
        )
        super().__init__(params, defaults)
    

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
    

    def _init_group(
        self,
        group,
        params,
        grads,
        n_buffer,
        z_buffer
    ):
        for p in group["params"]:
            if p.grad is None:
                continue

            params.append(p)
            grads.append(p.grad)
            state = self.state[p]
            if len(state) == 0:
                state["n"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                state["z"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
            n_buffer.append(state["n"])
            z_buffer.append(state["z"])

    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            buffers = [[] for _ in range(4)]
            self._init_group(group, *buffers)
            ftrl(
                *buffers,
                lr=group["lr"],
                alpha=group["alpha"],
                weight_shrink=group["weight_shrink"],
                weight_decay=group["weight_decay"],
                maximize=group["maximize"]
            )
        
        return loss


class FTRLAdam(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_shrink: float = 0, # L1 param
        weight_decay: float = 0, # L2 param
        *,
        maximize: bool = False,
    ) -> None:
        # TODO: 参数检查

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_shrink=weight_shrink,
            weight_decay=weight_decay,
            maximize=maximize,
        )
        super().__init__(params, defaults)
    

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
    

    def _init_group(
        self,
        group,
        params,
        grads,
        m_buffer,
        v_buffer,
        z_buffer,
        steps
    ):
        for p in group["params"]:
            if p.grad is None:
                continue

            params.append(p)
            grads.append(p.grad)
            state = self.state[p]
            if len(state) == 0:
                state["m"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                state["v"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                state["z"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                state["step"] = torch.zeros(
                    (), device=p.device
                )
            m_buffer.append(state["m"])
            v_buffer.append(state["v"])
            z_buffer.append(state["z"])
            steps.append(state["step"])

    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            buffers = [[] for _ in range(6)]
            self._init_group(group, *buffers)
            ftrl_adam(
                *buffers,
                lr=group["lr"],
                betas=group["betas"],
                weight_shrink=group["weight_shrink"],
                weight_decay=group["weight_decay"],
                maximize=group["maximize"]
            )
        
        return loss


def ftrl(
    params: list[Tensor],
    grads: list[Tensor],
    n_buffer: list[Tensor],
    z_buffer: list[Tensor],
    *,
    lr: float,
    alpha: float,
    weight_shrink: float,
    weight_decay: float,
    maximize: bool,
):
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        n = n_buffer[i]
        z = z_buffer[i]
        eta_old = (torch.sqrt(n) * alpha + 1) / lr if n.sum() > 0 else 0
        n.addcmul_(grad, grad, value=1)
        eta_new = (torch.sqrt(n) * alpha + 1) / lr
        sigma = eta_new - eta_old
        z.add_(grad).addcmul_(sigma, param, value=-1)
        param.set_(-torch.nn.functional.softshrink(z, weight_shrink) / (eta_new + weight_decay))


def ftrl_adam(
    params: list[Tensor],
    grads: list[Tensor],
    m_buffer: list[Tensor],
    v_buffer: list[Tensor],
    z_buffer: list[Tensor],
    steps: list[Tensor],
    *,
    lr: float,
    betas: tuple[float, float],
    weight_shrink: float,
    weight_decay: float,
    maximize: bool,
):
    beta1, beta2 = betas
    one_minus_beta1 = 1.0 - beta1
    one_minus_beta2 = 1.0 - beta2
    eps = 1.0e-8

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        m = m_buffer[i]
        v = v_buffer[i]
        z = z_buffer[i]
        step_t = steps[i]

        current_step = step_t.item()
        step_t.add_(1)
        beta1_pow_current = beta1 ** current_step
        beta2_pow_current = beta2 ** current_step
        beta1_pow_next = beta1_pow_current * beta1
        beta2_pow_next = beta2_pow_current * beta2

        eta_old = torch.sqrt_(v / ((1 - beta2_pow_current) if current_step > 0 else 1))
        m.mul_(beta1).add_(grad, alpha=(one_minus_beta1))
        v.mul_(beta2).addcmul_(grad, grad, value=one_minus_beta2)
        eta_new = torch.sqrt_(v / (1 - beta2_pow_next))
        z.add_(m, alpha=-(lr / (1 - beta1_pow_next))).addcmul_((eta_new - eta_old), param)
        torch.div(torch.nn.functional.softshrink(z, weight_shrink), eta_new.add_(weight_decay * lr + eps), out=param)

