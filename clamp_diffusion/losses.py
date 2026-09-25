from types import SimpleNamespace
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.func import functional_call

from .utils import compute_snr


def get_loss(model_pred, target, timesteps, noise_scheduler, snr_gamma):
    if snr_gamma is None:
        return F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    snr = compute_snr(timesteps, noise_scheduler)
    mse_loss_weights = (
        torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
    )
    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
    return loss.mean()


def gather_trainable_adapter_params(model) -> Dict[str, Tensor]:
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def rms_of_weights(params: Dict[str, Tensor], eps: float = 1e-12) -> Dict[str, Tensor]:
    return {k: torch.sqrt(torch.mean(p.detach() ** 2) + eps) for k, p in params.items()}


def flatten_param_dict(param_dict: Dict[str, Tensor]) -> Tensor:
    return torch.cat([p.flatten() for p in param_dict.values()])


def flatten_grad_list_layer_intact(grad_list: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    layer_wise = {}
    for grad_dict in grad_list:
        for layer_name, grad_tensor in grad_dict.items():
            layer_wise.setdefault(layer_name, []).append(grad_tensor.flatten())
    return {k: torch.stack(v, dim=1) for k, v in layer_wise.items()}


def safe_zero(device) -> Tensor:
    return torch.tensor(0.0, dtype=torch.double, device=device)


def functional_forward(model, param_override: Dict[str, Tensor], noisy_latents, timesteps, encoder_hidden_states, **fw_kwargs):
    return functional_call(model, param_override, (noisy_latents, timesteps, encoder_hidden_states), kwargs=fw_kwargs).sample


def sgd_update(params: Dict[str, Tensor], grads: List[Tensor], lr: float) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    updated_params, step = {}, {}
    for (name, p), g in zip(params.items(), grads):
        if g is None:
            updated_params[name] = p
            step[name] = torch.zeros_like(p)
        else:
            updated_params[name] = p - lr * g
            step[name] = -lr * g
    return updated_params, step


def turn_list2dict(param_list, keys) -> Dict[str, Tensor]:
    return {k: v for k, v in zip(keys, param_list)}


def get_normalized_by_layer(params: Dict[str, Tensor], eps: float = 1e-12) -> Dict[str, Tensor]:
    return {k: v / (torch.norm(v) + eps) for k, v in params.items()}


def inner_k_step_stateless_batch(args, model, batch, lora_params: Dict[str, Tensor], lr: float, K: int, low_memory: bool = False):
    current_params = lora_params
    keys = list(current_params.keys())
    create_graph = not low_memory

    u_list, g_list, theta_list = [], [], []
    u_list_norm, g_list_norm = [], []
    loss0, grad0, lossK, gradK = None, None, None, None

    for t in range(K):
        out_t = functional_forward(
            model, {**dict(model.named_parameters()), **current_params},
            noisy_latents=batch["noisy_latents"], timesteps=batch["timesteps"],
            encoder_hidden_states=batch["encoder_hidden_states"],
        )
        loss_t = get_loss(out_t, batch["target"], batch["timesteps"], batch["scheduler"], args.snr_gamma) * args.scale_loss_inner

        grads_t = torch.autograd.grad(
            loss_t, [current_params[k] for k in keys],
            create_graph=create_graph, retain_graph=True, allow_unused=False,
        )
        grads_t_dict = turn_list2dict(grads_t, keys)

        proposed_params, step = sgd_update(current_params, grads_t, lr=lr)
        current_params = proposed_params

        u_list.append(step)
        g_list.append(grads_t_dict)
        u_list_norm.append(get_normalized_by_layer(step))
        g_list_norm.append(get_normalized_by_layer(grads_t_dict))
        theta_list.append(proposed_params)
        if t == 0:
            loss0, grad0 = loss_t, grads_t
        if t == K - 1:
            lossK, gradK = loss_t, grads_t

    out_tp1 = functional_forward(
        model, {**dict(model.named_parameters()), **current_params},
        noisy_latents=batch["noisy_latents"], timesteps=batch["timesteps"],
        encoder_hidden_states=batch["encoder_hidden_states"],
    )
    loss_tp1 = get_loss(out_tp1, batch["target"], batch["timesteps"], batch["scheduler"], args.snr_gamma)
    grads_tp1 = torch.autograd.grad(
        loss_tp1, [current_params[k] for k in keys],
        create_graph=create_graph, retain_graph=True, allow_unused=False,
    )
    theta_kplus1, _ = sgd_update(current_params, grads_tp1, lr=lr)

    return SimpleNamespace(
        u_list=u_list, g_list=g_list, u_list_norm=u_list_norm, g_list_norm=g_list_norm,
        theta_list=theta_list, thetaK=proposed_params, loss0=loss0, grad0=grad0,
        lossK=lossK, gradK=gradK, theta_kplus1=theta_kplus1,
    )


def ortho_from_cols(M: Tensor, r: int, drop_eps: float = 1e-12) -> Tensor:
    if M.numel() == 0 or M.shape[1] == 0:
        return torch.zeros(M.shape[0], 0, dtype=M.dtype, device=M.device)
    U, S, _ = torch.linalg.svd(M, full_matrices=False)
    keep = min(r, (S > drop_eps).sum().item())
    return U[:, :keep] if keep > 0 else torch.zeros(M.shape[0], 0, dtype=M.dtype, device=M.device)


def subspace_subtract_reorthonormalize(U1: Tensor, U2: Tensor) -> Tensor:
    if U1.shape[1] > 0 and U2.shape[1] > 0:
        U1_only = U1 - U2 @ (U2.T @ U1)
        if U1_only.numel() > 0 and U1_only.shape[1] > 0:
            U1_only_norm, _ = torch.linalg.qr(U1_only, mode="reduced")
            if U1_only_norm.numel() > 0 and U1_only_norm.shape[1] > 0:
                return U1_only_norm
    return U1


def subspaces_and_projectors_per_layer(g_norm_H, g_norm_G, rH, rP):
    layer_projectors = {}
    gH_flat = flatten_grad_list_layer_intact(g_norm_H)
    gG_flat = flatten_grad_list_layer_intact(g_norm_G)
    for layer_name in gH_flat.keys():
        Uh = ortho_from_cols(gH_flat[layer_name], rH)
        Ug = ortho_from_cols(gG_flat[layer_name], rP)
        U = subspace_subtract_reorthonormalize(Uh, Ug)
        layer_projectors[layer_name] = (lambda v, U_mat=U: U_mat @ (U_mat.T @ v))
    return layer_projectors


def project_onto_subspace(flat_tensor: Tensor, projector, normalize: bool = False) -> Tensor:
    if projector is None:
        return flat_tensor
    v_proj = projector(flat_tensor)
    if normalize:
        v_proj = v_proj / (torch.norm(v_proj) + 1e-12)
    return v_proj


def project_onto_subspace_per_layer(dict_tensors: Dict[str, Tensor], projector, normalize: bool = False) -> Tensor:
    if projector is None:
        return flatten_param_dict(dict_tensors)
    out = []
    for layer_name, tensor in dict_tensors.items():
        out.append(project_onto_subspace(tensor.flatten(), projector.get(layer_name), normalize=normalize))
    return torch.cat(out)


def unflatten_and_scale(flat_tensor: Tensor, scales: Dict[str, Tensor], param_dict: Dict[str, Tensor], eps: float = 1e-12) -> Dict[str, Tensor]:
    new_param_dict = {}
    idx = 0
    for k, p in param_dict.items():
        n = p.numel()
        new_param_dict[k] = flat_tensor[idx:idx + n].view(p.shape) * scales[k] * eps
        idx += n
    return new_param_dict


def plateau_curvature_loss(args, model, harm_batch, scales, params_k, u_normalized_flat, plateau_eps, lk, kappa_min=3.0, kappa_max=10.0):
    delta_params = unflatten_and_scale(u_normalized_flat, scales, params_k, eps=plateau_eps)

    out_plus = functional_forward(
        model, {**dict(model.named_parameters()), **{k: params_k[k] + delta_params[k] for k in params_k}},
        noisy_latents=harm_batch["noisy_latents"], timesteps=harm_batch["timesteps"],
        encoder_hidden_states=harm_batch["encoder_hidden_states"],
    )
    L_plus = get_loss(out_plus, harm_batch["target"], harm_batch["timesteps"], harm_batch["scheduler"], args.snr_gamma)

    out_minus = functional_forward(
        model, {**dict(model.named_parameters()), **{k: params_k[k] - delta_params[k] for k in params_k}},
        noisy_latents=harm_batch["noisy_latents"], timesteps=harm_batch["timesteps"],
        encoder_hidden_states=harm_batch["encoder_hidden_states"],
    )
    L_minus = get_loss(out_minus, harm_batch["target"], harm_batch["timesteps"], harm_batch["scheduler"], args.snr_gamma)

    delta_sq_sum = torch.stack([delta_params[k].pow(2).sum() for k in delta_params]).sum()
    global_delta = torch.clamp(torch.sqrt(delta_sq_sum + 1e-12), min=1e-3, max=1e-2)

    curvature_fd = (L_plus + L_minus - 2 * lk) / (global_delta ** 2 + 1e-12)
    curvature_fd = torch.clamp(curvature_fd, min=0.0)

    lower = F.softplus(kappa_min - curvature_fd)
    return lower, {
        "plateau_lower": lower.detach().cpu().item(),
        "plateau_global_delta": global_delta.detach().cpu().item(),
        "plateau_curvature": curvature_fd.detach().cpu().item(),
    }


def normalize_flat(v_flat: Tensor, eps: float = 1e-12) -> Tensor:
    return v_flat / (torch.linalg.vector_norm(v_flat) + eps)


def random_harmful_dir(params: Dict[str, Tensor]) -> Tensor:
    parts = [torch.randn_like(p).reshape(-1) for p in params.values()]
    return normalize_flat(torch.cat(parts))


def _contractivity_estimation_impl(args, model, harm_batch, scales, v_normalized_flat_dict, contractivity_eps, inner_lr, theta_k, theta_kplus1, power_iter, create_graph):
    c_max, c_name = torch.tensor(0.0), None
    for opt, v in v_normalized_flat_dict.items():
        for _ in range(power_iter):
            delta_params = unflatten_and_scale(v, scales, theta_k, eps=contractivity_eps)
            theta_plus = {**dict(model.named_parameters()), **{k: theta_k[k] + delta_params[k] for k in theta_k}}

            out_plus = functional_forward(
                model, theta_plus,
                noisy_latents=harm_batch["noisy_latents"], timesteps=harm_batch["timesteps"],
                encoder_hidden_states=harm_batch["encoder_hidden_states"],
            )
            loss_plus = get_loss(out_plus, harm_batch["target"], harm_batch["timesteps"], harm_batch["scheduler"], args.snr_gamma)

            grads_plus = torch.autograd.grad(
                loss_plus, [theta_plus[k] for k in theta_k],
                create_graph=create_graph, retain_graph=True, allow_unused=False,
            )
            sgd_updated_params, _ = sgd_update(theta_k, grads_plus, lr=inner_lr)

            delta_sq_sum = torch.stack([delta_params[k].pow(2).sum() for k in delta_params]).sum()
            global_delta_norm = torch.clamp(torch.sqrt(delta_sq_sum + 1e-12), min=1e-3, max=1e-2)

            w = {k: (sgd_updated_params[k] - theta_kplus1[k]) / global_delta_norm for k in theta_kplus1}
            contractivity = torch.sqrt(torch.stack([wi.pow(2).sum() for wi in w.values()]).sum() + 1e-12)
            v = flatten_param_dict({k: wv / contractivity for k, wv in w.items()})

        if contractivity > c_max:
            c_max, c_name = contractivity, opt

    return c_max, {"contractivity_max": c_max.detach().cpu().item(), "contractivity_name": c_name}


def contractivity_estimation(args, model, harm_batch, scales, v_normalized_flat_dict, contractivity_eps, inner_lr, theta_k, theta_kplus1, power_iter=2, low_memory=False):
    return _contractivity_estimation_impl(args, model, harm_batch, scales, v_normalized_flat_dict, contractivity_eps, inner_lr, theta_k, theta_kplus1, power_iter, create_graph=not low_memory)


def _lipschitz_estimation_impl(args, model, harm_batch, scales, thetaK, u_normalized_flat, lhat_eps, gradK, create_graph):
    delta_params = unflatten_and_scale(u_normalized_flat, scales, thetaK, eps=lhat_eps)

    out_deltaK = functional_forward(
        model, {**dict(model.named_parameters()), **{k: thetaK[k] + delta_params[k] for k in thetaK}},
        noisy_latents=harm_batch["noisy_latents"], timesteps=harm_batch["timesteps"],
        encoder_hidden_states=harm_batch["encoder_hidden_states"],
    )
    loss_deltaK = get_loss(out_deltaK, harm_batch["target"], harm_batch["timesteps"], harm_batch["scheduler"], args.snr_gamma)

    grads_deltaK = torch.autograd.grad(
        loss_deltaK, [thetaK[k] for k in thetaK],
        create_graph=create_graph, retain_graph=True, allow_unused=False,
    )

    delta_sq_sum = torch.stack([delta_params[k].pow(2).sum() for k in delta_params]).sum()
    global_delta_norm = torch.clamp(torch.sqrt(delta_sq_sum + 1e-12), min=1e-3, max=1e-2)

    diff_norm = torch.norm(torch.cat([(g1 - g2).reshape(-1) for g1, g2 in zip(grads_deltaK, gradK)]))
    lipschitz = diff_norm / (global_delta_norm + 1e-12)

    return lipschitz, {
        "lipschitz_global_delta": global_delta_norm.detach().cpu().item(),
        "lipschitz": lipschitz.detach().cpu().item(),
    }


def lipschitz_estimation(args, model, harm_batch, scales, thetaK, u_normalized_flat, lhat_eps, gradK, low_memory=False):
    return _lipschitz_estimation_impl(args, model, harm_batch, scales, thetaK, u_normalized_flat, lhat_eps, gradK, create_graph=not low_memory)
