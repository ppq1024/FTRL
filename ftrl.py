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
        sparse: float = 0, # L1 param
        weight_decay: float = 0, # L2 param
        *,
        maximize: bool = False,
    ) -> None:
        # TODO: 参数检查

        defaults = dict(
            lr=lr,
            alpha=alpha,
            sparse=sparse,
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
                sparse=group["sparse"],
                weight_decay=group["weight_decay"],
                maximize=group["maximize"]
            )
        
        return loss


class FTRLAdam(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        alpha: float = 1.0,
        betas: tuple[float, float] = (0.9, 0.999),
        sparse: float = 0, # L1 param
        weight_decay: float = 0, # L2 param
        *,
        maximize: bool = False,
    ) -> None:
        # TODO: 参数检查

        defaults = dict(
            lr=lr,
            alpha=alpha,
            betas=betas,
            sparse=sparse,
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
        n_buffer,
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
                state["n"] = torch.zeros_like(
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
            n_buffer.append(state["n"])
            z_buffer.append(state["z"])
            steps.append(state["step"])

    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            buffers = [[] for _ in range(7)]
            self._init_group(group, *buffers)
            ftrl_adam(
                *buffers,
                lr=group["lr"],
                alpha=group["alpha"],
                betas=group["betas"],
                sparse=group["sparse"],
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
    sparse: float,
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
        param.set_(-torch.nn.functional.softshrink(z, sparse) / (eta_new + weight_decay))


def ftrl_adam(
    params: list[Tensor],
    grads: list[Tensor],
    m_buffer: list[Tensor],
    v_buffer: list[Tensor],
    n_buffer: list[Tensor],
    z_buffer: list[Tensor],
    steps: list[Tensor],
    *,
    lr: float,
    alpha: float,
    betas: tuple[float, float],
    sparse: float,
    weight_decay: float,
    maximize: bool,
):
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        m = m_buffer[i]
        v = v_buffer[i]
        n = n_buffer[i]
        z = z_buffer[i]
        step_t = steps[i]
        step_t.add_(1)
        m.mul_(betas[0]).add_(grad, alpha=(1 - betas[0]))
        v.mul_(betas[1]).addcmul_(grad, grad, value=(1 - betas[1]))
        m_hat = m / (1 - betas[0] ** step_t)
        v_hat = v / (1 - betas[1] ** step_t)
        eta_old = (torch.sqrt(n) * alpha + 1) / lr if step_t > 0 else 0
        n.add_(v_hat)
        eta_new = (torch.sqrt(n) * alpha + 1) / lr
        sigma = eta_new - eta_old
        z.add_(m_hat / torch.sqrt(v_hat + 1.0e-8), alpha=1 / (1 - betas[0] ** step_t)).addcmul_(sigma, param, value=-1)
        param.set_(-torch.nn.functional.softshrink(z, sparse) / (eta_new + weight_decay))

