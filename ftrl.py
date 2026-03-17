import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT
from typing import Any, Callable, Optional, Union, overload, override

__all__ = ["FTRL", "FTRLAdam"]


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
        foreach: bool = True
    ) -> None:
        # TODO: 参数检查

        defaults = dict(
            lr=lr,
            alpha=alpha,
            weight_shrink=weight_shrink,
            weight_decay=weight_decay,
            maximize=maximize,
            foreach=foreach
        )
        super().__init__(params, defaults)
    

    # TODO
    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
    

    def _init_group(
        self,
        group: dict[str, Any],
        params: list[Tensor],
        grads: list[Tensor],
        n_buffer: list[Tensor],
        z_buffer: list[Tensor],
        steps: list[Tensor]
    ):
        for p in group["params"]:
            p: Tensor
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
                state["step"] = torch.zeros(
                    (), device=p.device
                )
            n_buffer.append(state["n"])
            z_buffer.append(state["z"])
            steps.append(state["step"])

    @overload
    def step(self, closure: None = None) -> None: ...


    @overload
    def step(self, closure: Callable[[], float]) -> float: ...


    @torch.no_grad()
    @override
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            buffers: list[list[Tensor]] = [[] for _ in range(5)]
            self._init_group(group, *buffers)
            ftrl(
                *buffers,
                lr=group["lr"],
                alpha=group["alpha"],
                weight_shrink=group["weight_shrink"],
                weight_decay=group["weight_decay"],
                maximize=group["maximize"],
                foreach=group["foreach"]
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
        foreach: bool = True
    ) -> None:
        # TODO: 参数检查

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_shrink=weight_shrink,
            weight_decay=weight_decay,
            maximize=maximize,
            foreach=foreach
        )
        super().__init__(params, defaults)
    

    # TODO
    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
    

    def _init_group(
        self,
        group: dict[str, Any],
        params: list[Tensor],
        grads: list[Tensor],
        m_buffer: list[Tensor],
        v_buffer: list[Tensor],
        z_buffer: list[Tensor],
        steps: list[Tensor]
    ):
        for p in group["params"]:
            p: Tensor
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


    @overload
    def step(self, closure: None = None) -> None: ...


    @overload
    def step(self, closure: Callable[[], float]) -> float: ...
    
    @torch.no_grad()
    @override
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            buffers: list[list[Tensor]] = [[] for _ in range(6)]
            self._init_group(group, *buffers)
            ftrl_adam(
                *buffers,
                lr=group["lr"],
                betas=group["betas"],
                weight_shrink=group["weight_shrink"],
                weight_decay=group["weight_decay"],
                maximize=group["maximize"],
                foreach=group["foreach"]
            )
        
        return loss


def ftrl_single(
    params: list[Tensor],
    grads: list[Tensor],
    n_buffer: list[Tensor],
    z_buffer: list[Tensor],
    steps: list[Tensor],
    *,
    lr: float,
    alpha: float,
    weight_shrink: float,
    weight_decay: float,
    maximize: bool
):
    weight_decay *= lr
    weight_shrink *= lr
    factor_grad = lr if maximize else -lr
    
    for param, grad, n, z, step in zip(params, grads, n_buffer, z_buffer, steps):
        eta_old = (torch.sqrt(n) * alpha + 1) * torch.sign(step)
        step.add_(1)
        n.addcmul_(grad, grad)
        eta_new = (torch.sqrt(n) * alpha + 1)
        factor_p = eta_new - eta_old - weight_decay
        z.addcmul_(param, factor_p).add_(grad, alpha=factor_grad)
        torch.div(torch.nn.functional.softshrink(z, weight_shrink), eta_new + weight_decay, out=param)


def ftrl_foreach(
    params: list[Tensor],
    grads: list[Tensor],
    n_buffer: list[Tensor],
    z_buffer: list[Tensor],
    steps: list[Tensor],
    *,
    lr: float,
    alpha: float,
    weight_shrink: float,
    weight_decay: float,
    maximize: bool
):
    weight_decay *= lr
    weight_shrink *= lr
    factor_grad = lr if maximize else -lr

    eta_old = torch._foreach_sqrt(n_buffer)
    torch._foreach_mul_(eta_old, alpha)
    torch._foreach_add_(eta_old, 1)
    torch._foreach_mul_(eta_old, torch._foreach_sign(steps))
    torch._foreach_add_(steps, 1)

    torch._foreach_addcmul_(n_buffer, grads, grads)
    eta_new = torch._foreach_sqrt(n_buffer)
    torch._foreach_mul_(eta_new, alpha)
    torch._foreach_add_(eta_new, 1)

    factor_p = torch._foreach_sub(eta_new, eta_old)
    if weight_decay > 0:
        torch._foreach_add_(factor_p, -weight_decay)

    torch._foreach_addcmul_(z_buffer, params, factor_p)
    torch._foreach_add_(z_buffer, grads, alpha=factor_grad)
    
    for param, z, eta_new in zip(params, z_buffer, eta_new):
        torch.div(torch.nn.functional.softshrink(z, weight_shrink), eta_new + weight_decay, out=param)


def ftrl(
    params: list[Tensor],
    grads: list[Tensor],
    n_buffer: list[Tensor],
    z_buffer: list[Tensor],
    steps: list[Tensor],
    *,
    lr: float,
    alpha: float,
    weight_shrink: float,
    weight_decay: float,
    maximize: bool,
    foreach: bool
):
    func = ftrl_foreach if foreach else ftrl_single
    func(
        params,
        grads,
        n_buffer,
        z_buffer,
        steps,
        lr=lr,
        alpha=alpha,
        weight_shrink=weight_shrink,
        weight_decay=weight_decay,
        maximize=maximize
    )

def ftrl_adam_single(
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
    maximize: bool
):
    beta1, beta2 = betas
    one_minus_beta1 = 1.0 - beta1
    one_minus_beta2 = 1.0 - beta2
    weight_decay *= lr
    weight_shrink *= lr
    factor_grad = lr if maximize else -lr
    eps = 1.0e-8
    
    for item in zip(params, grads, m_buffer, v_buffer, z_buffer, steps):
        item: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
        param, grad, m, v, z, step_t = item
        
        current_step = step_t.item()
        step_t.add_(1)
        beta2_pow_current = beta2 ** current_step
        beta2_pow_next = beta2_pow_current * beta2
        factor_v_current = 1 / (1 - beta2_pow_current + eps)
        factor_v_next = 1 / (1 - beta2_pow_next)
        
        eta_old = torch.sqrt(v * factor_v_current)
        m.mul_(beta1).add_(grad, alpha=one_minus_beta1)
        v.mul_(beta2).addcmul_(grad, grad, value=one_minus_beta2)
        eta_new = torch.sqrt(v * factor_v_next)
        factor_m = factor_grad / (1 - beta1 ** (current_step + 1))
        factor_p = eta_new - eta_old - weight_decay
        z.add_(m, alpha=factor_m).addcmul_(param, factor_p)
        torch.div(torch.nn.functional.softshrink(z, weight_shrink), eta_new + weight_decay + eps, out=param)


def ftrl_adam_foreach(
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
    maximize: bool
):
    beta1, beta2 = betas
    one_minus_beta1 = 1.0 - beta1
    one_minus_beta2 = 1.0 - beta2
    weight_decay *= lr
    weight_shrink *= lr
    factor_grad = lr if maximize else -lr
    eps = 1.0e-8

    factor_v_current = torch._foreach_pow(beta2, steps)
    torch._foreach_add_(steps, 1)
    factor_v_next = torch._foreach_mul(factor_v_current, beta2)
    torch._foreach_mul_(factor_v_current, -1.0)
    torch._foreach_add_(factor_v_current, 1.0)
    torch._foreach_add_(factor_v_current, eps)
    torch._foreach_reciprocal_(factor_v_current)
    torch._foreach_mul_(factor_v_next, -1.0)
    torch._foreach_add_(factor_v_next, 1.0)
    torch._foreach_reciprocal_(factor_v_next)

    factor_m = torch._foreach_pow(beta1, steps)
    torch._foreach_mul_(factor_m, -1.0)
    torch._foreach_add_(factor_m, 1.0)
    torch._foreach_reciprocal_(factor_m)
    torch._foreach_mul_(factor_m, factor_grad)

    eta_old = torch._foreach_mul(v_buffer, factor_v_current)
    torch._foreach_sqrt_(eta_old)
    torch._foreach_mul_(m_buffer, beta1)
    torch._foreach_add_(m_buffer, grads, alpha=one_minus_beta1)
    torch._foreach_mul_(v_buffer, beta2)
    torch._foreach_addcmul_(v_buffer, grads, grads, value=one_minus_beta2)
    eta_new = torch._foreach_mul(v_buffer, factor_v_next)
    torch._foreach_sqrt_(eta_new)
    factor_p = torch._foreach_sub(eta_new, eta_old)
    if weight_decay > 0:
        torch._foreach_add_(factor_p, -weight_decay)

    torch._foreach_addcmul_(z_buffer, params, factor_p)
    torch._foreach_addcmul_(z_buffer, m_buffer, factor_m)
    

    for param, eta_new_i, z in zip(params, eta_new, z_buffer):
        torch.div(torch.nn.functional.softshrink(z, weight_shrink), eta_new_i + weight_decay + eps, out=param)


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
    foreach: bool
):
    func = ftrl_adam_foreach if foreach else ftrl_adam_single
    func(
        params,
        grads,
        m_buffer,
        v_buffer,
        z_buffer,
        steps,
        lr=lr,
        betas=betas,
        weight_shrink=weight_shrink,
        weight_decay=weight_decay,
        maximize=maximize
    )
