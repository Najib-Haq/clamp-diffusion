import contextlib
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from .utils import compute_snr

EPS = 1e-12


def get_loss(model_pred, target, timesteps, noise_scheduler, snr_gamma):
    if snr_gamma is None:
        return F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    snr = compute_snr(timesteps, noise_scheduler)
    mse_loss_weights = torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
    return loss.mean()


def trainable_params(model) -> Dict[str, Tensor]:
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def norm(vector: Tensor) -> Tensor:
    return torch.linalg.vector_norm(vector) + EPS


def unit(vector: Tensor) -> Tensor:
    return vector / norm(vector)


@contextlib.contextmanager
def exact_fp32():
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.autocast(device_type="cuda", enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


class Probe:
    def __init__(self, config, model, batch, params):
        self.config = config
        self.model = model
        self.params = params
        self.keys = list(params)
        self.shapes = [params[k].shape for k in self.keys]
        self.sizes = [params[k].numel() for k in self.keys]
        size = batch["noisy_latents"].shape[0] if config.probe_batch_size <= 0 else config.probe_batch_size
        self.batch = {
            k: (v[:size].float() if torch.is_tensor(v) and v.is_floating_point() else v[:size] if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        flat = torch.cat([params[k].detach().float().reshape(-1) for k in self.keys])
        floor = float(torch.sqrt(flat.pow(2).mean() + EPS))
        self.scale = torch.cat([
            torch.full((n,), max(float(torch.sqrt(params[k].detach().float().pow(2).mean() + EPS)), floor), device=flat.device)
            for k, n in zip(self.keys, self.sizes)
        ])

    def unflatten(self, flat):
        return {k: part.view(shape) for k, part, shape in zip(self.keys, torch.split(flat, self.sizes), self.shapes)}

    def loss_grad(self, theta):
        live = [self.params[k] for k in self.keys]
        saved = [p.data.clone() for p in live]
        for p, part in zip(live, self.unflatten(theta.detach()).values()):
            p.data.copy_(part)
        try:
            pred = self.model(
                self.batch["noisy_latents"], self.batch["timesteps"], self.batch["encoder_hidden_states"],
            ).sample
            loss = get_loss(pred, self.batch["target"], self.batch["timesteps"], self.batch["scheduler"], self.config.snr_gamma)
            grads = torch.autograd.grad(loss, live)
        finally:
            for p, s in zip(live, saved):
                p.data.copy_(s)
        return loss.detach(), torch.cat([g.reshape(-1).float() for g in grads]).detach()

    def scaled(self, vector):
        return unit(vector / self.scale)

    def step(self, direction, eps):
        shift = direction * eps * self.scale
        return shift, norm(shift)

    def hvp(self, theta, grad, vector):
        shift, size = self.step(self.scaled(vector), self.config.hvp_eps)
        return (self.loss_grad(theta + shift)[1] - grad) / size


def _secant(probe, theta, grad, direction, lr):
    shift, size = probe.step(direction, probe.config.contractivity_eps)
    _, shifted_grad = probe.loss_grad(theta + shift)
    difference = shift - lr * (shifted_grad - grad)
    return difference, norm(difference) / size, shift, size, shifted_grad


def contractivity(probe, theta, grad, candidates, lr):
    best = None
    for index, candidate in enumerate(candidates):
        direction = candidate
        for _ in range(probe.config.contractivity_power_iters):
            difference = _secant(probe, theta, grad, direction, lr)[0]
            direction = probe.scaled(difference)
        difference, value, shift, size, shifted_grad = _secant(probe, theta, grad, direction, lr)
        if best is None or value > best[1]:
            best = (difference, value, shift, size, shifted_grad, index)
    difference, value, shift, size, shifted_grad, index = best
    w = unit(difference)
    gradient = -lr * (probe.hvp(theta + shift, shifted_grad, w) - probe.hvp(theta, grad, w)) / size
    return value, gradient, index


def lipschitz(probe, theta, grad, direction):
    shift, size = probe.step(direction, probe.config.lhat_eps)
    _, shifted_grad = probe.loss_grad(theta + shift)
    residual = shifted_grad - grad
    r = unit(residual)
    gradient = (probe.hvp(theta + shift, shifted_grad, r) - probe.hvp(theta, grad, r)) / size
    return norm(residual) / size, gradient


def curvature(probe, theta, loss, grad, direction):
    shift, size = probe.step(direction, probe.config.plateau_eps)
    loss_plus, grad_plus = probe.loss_grad(theta + shift)
    loss_minus, grad_minus = probe.loss_grad(theta - shift)
    value = (loss_plus + loss_minus - 2.0 * loss) / size.pow(2)
    gradient = (grad_plus + grad_minus - 2.0 * grad) / size.pow(2)
    return value, gradient


def immunization_losses(config, model, harm_batch):
    params = trainable_params(model)
    theta_live = torch.cat([p.reshape(-1).float() for p in params.values()])
    lr = config.inner_sgd_lr
    K = config.inner_k_steps

    def surrogate(value, gradient):
        linear = torch.dot(theta_live, gradient)
        return value + linear - linear.detach()

    with exact_fp32():
        probe = Probe(config, model, harm_batch, params)
        theta = theta_live.detach()
        anchors, losses, grads = [], [], []
        for _ in range(K + 1):
            loss, grad = probe.loss_grad(theta)
            anchors.append(theta)
            losses.append(loss)
            grads.append(grad)
            theta = theta - lr * grad

        kappas = [
            curvature(probe, anchors[t], losses[t], grads[t], probe.scaled(-lr * grads[t]))
            for t in range(K + 1)
        ]

        theta_K, grad_K = anchors[K], grads[K]
        candidates = [probe.scaled(-lr * grad_K), probe.scaled(grad_K)]
        if config.contractivity_random_probe:
            candidates.append(unit(torch.randn_like(grad_K)))
        c_value, c_gradient, c_choice = contractivity(probe, theta_K, grad_K, candidates, lr)
        L_value, L_gradient = lipschitz(probe, theta_K, grad_K, candidates[0])
        g_norm_gradient = probe.hvp(theta_K, grad_K, unit(grad_K))

    loss_start = surrogate(losses[0], grads[0])
    loss_final = surrogate(losses[K], grads[K])
    kappa = [surrogate(v, g) for v, g in kappas]
    c_hat = surrogate(c_value, c_gradient)
    L_hat = surrogate(L_value, L_gradient)
    g_norm = surrogate(norm(grad_K), g_norm_gradient)
    u_norm = lr * g_norm

    delta_act = config.lambda_actual * (loss_start - loss_final)
    floor = torch.tensor(1.0 - config.c_clip, device=c_hat.device)
    B_tail = torch.clamp(u_norm / torch.maximum(1.0 - c_hat, floor), max=config.tail_clip)
    delta_tail = config.lambda_estimated * (B_tail * g_norm + 0.5 * L_hat * B_tail.pow(2))

    kappa_min = torch.tensor(config.kappa_min, device=loss_start.device)
    curv_terms = torch.stack([F.softplus(kappa_min - k) for k in kappa])
    terms = {
        "L_long": F.relu(delta_act + delta_tail - config.long_margin),
        "L_contract": F.softplus(c_hat - config.c_target),
        "L_plateau": curv_terms.sum() if config.curvature_reduction == "sum" else curv_terms.mean(),
        "L_inverse": -loss_start,
    }
    stats = {
        "delta_act": delta_act, "delta_tail": delta_tail, "c_hat": c_hat, "c_choice": float(c_choice),
        "L_hat": L_hat, "B_tail": B_tail, "u_K_norm": u_norm, "g_K_norm": g_norm,
        **{f"kappa_{t}": k for t, k in enumerate(kappa)},
        **{f"loss_H{t}": l for t, l in enumerate(losses)},
    }
    return terms, {k: v.detach() if torch.is_tensor(v) else v for k, v in stats.items()}
