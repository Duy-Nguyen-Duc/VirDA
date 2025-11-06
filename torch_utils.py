import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np


from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple, Any, List, Optional

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)

def freeze_layers(layers: list[nn.Module]):
    """
    Freeze layers listed in the model
    """
    for layer in layers:
        layer.eval()
        for param in layer.parameters():
            param.requires_grad = False


def compute_hard_alpha(u: torch.Tensor, t_u: float):
    """
    Compute hard alpha.

    Args:
        u (torch.Tensor): filtering tensor
        t_u (float): threshold

    """
    mask = (u <= t_u).float()
    w = torch.exp(-u) * mask
    return w / (w.sum() + 1e-12) * u.numel()


def compute_soft_alpha(u: torch.Tensor):
    w = torch.exp(-u)
    return w / (w.sum() + 1e-12) * u.numel()


def compute_soft_alpha_anneal(u, step, total_steps, min_temp=0.1, max_temp=1.0):
    frac = step / float(total_steps)
    T = max_temp * (1 - frac) + min_temp * frac
    w = torch.exp(-u / T)
    w = w / (w.max().clamp(min=1e-6))
    return w


def decay_thresholds(thres_start, thres_end, total_steps, method="exp"):
    if method == "exp":
        t = np.linspace(0, 1, total_steps)
        values = thres_start * (thres_end / thres_start) ** t
    elif method == "log":
        t = np.logspace(0, 1, total_steps, base=10)
        t = (t - t.min()) / (t.max() - t.min())
        values = thres_start - (thres_start - thres_end) * t
    else:
        raise ValueError("Invalid method. Use 'exp' or 'log'.")
    return list(values)


@dataclass
class FlopResult:
    total_flops: int
    by_module: Dict[str, int]
    by_class: Dict[str, int]

def _prod(xs: List[int]) -> int:
    p = 1
    for v in xs:
        p *= int(v)
    return p

def _strip_wrapper_prefixes(name: str) -> str:
    parts = name.split(".")
    if parts and parts[0] in {"module", "model", "net"} and len(parts) > 1:
        return ".".join(parts[1:])
    return name

def _top_level_component_name(name: str) -> str:
    clean = _strip_wrapper_prefixes(name)
    return clean.split(".", 1)[0]

def _conv_macs_adds(mod: nn.modules.conv._ConvNd, x: torch.Tensor, y: torch.Tensor, include_bias: bool) -> Tuple[int, int]:
    # y: (N, Cout, *spatial_out)
    n, cout, *out_spatial = y.shape
    out_elems = int(n) * int(cout) * _prod(list(map(int, out_spatial)))
    cin = int(mod.in_channels)
    groups = int(mod.groups)
    kernel_muladds = _prod(list(map(int, mod.kernel_size))) * (cin // groups)
    macs = out_elems * kernel_muladds
    adds = out_elems if (include_bias and mod.bias is not None) else 0
    return int(macs), int(adds)

def _linear_macs_adds(mod: nn.Linear, x: torch.Tensor, y: torch.Tensor, include_bias: bool) -> Tuple[int, int]:
    # y: (..., out_features)
    out_shape = list(map(int, y.shape))
    out_elems = _prod(out_shape[:-1]) * out_shape[-1]
    in_features = int(mod.in_features)
    out_features = int(mod.out_features)
    num_vecs = out_elems // out_features
    macs = num_vecs * in_features * out_features
    adds = out_elems if (include_bias and mod.bias is not None) else 0
    return int(macs), int(adds)

def _mha_macs_adds(mod: nn.MultiheadAttention,
                   inputs: Tuple[Any, ...],
                   output: Tuple[torch.Tensor, Optional[torch.Tensor]],
                   include_bias: bool) -> Tuple[int, int]:
    # Handle batch_first and optional K/V
    query = inputs[0]
    key = inputs[1] if len(inputs) > 1 and inputs[1] is not None else query
    value = inputs[2] if len(inputs) > 2 and inputs[2] is not None else key

    def get_b_l_e(t: torch.Tensor) -> Tuple[int, int, int]:
        if mod.batch_first:
            b, l, e = t.shape
        else:
            l, b, e = t.shape
        return int(b), int(l), int(e)

    Bq, Lq, E = get_b_l_e(query)
    Bk, Lk, Ek = get_b_l_e(key)
    Bv, Lv, Ev = get_b_l_e(value)
    assert E == Ek == Ev, "embed_dim must match for MHA MACs accounting"

    H = int(mod.num_heads)
    Dh = E // H

    macs = 0
    adds = 0

    macs += (Bq * Lq * E * E)
    macs += (Bk * Lk * E * E)
    macs += (Bv * Lv * E * E)
    if include_bias:
        adds += (Bq * Lq + Bk * Lk + Bv * Lv) * E

    macs += (Bq * H * Lq * Dh * Lk)
    macs += (Bq * H * Lq * Lk * Dh)
    macs += (Bq * Lq * E * E)
    if include_bias and getattr(mod, "out_proj", None) is not None and mod.out_proj.bias is not None:
        adds += Bq * Lq * E

    return int(macs), int(adds)

def _attach_hooks(model: nn.Module, include_bias: bool):
    handlers = []
    macs_by_module: Dict[str, int] = defaultdict(int)
    adds_by_module: Dict[str, int] = defaultdict(int)
    name_by_module: Dict[nn.Module, str] = {}
    class_by_module: Dict[nn.Module, str] = {}

    for name, m in model.named_modules():
        if name == "":
            continue
        name_by_module[m] = name
        class_by_module[m] = m.__class__.__name__

        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            def hook(mod, inp, out, *, _m=m):
                x = inp[0]
                macs, adds = _conv_macs_adds(_m, x, out, include_bias)
                macs_by_module[name_by_module[_m]] += macs
                adds_by_module[name_by_module[_m]] += adds
            handlers.append(m.register_forward_hook(hook))

        elif isinstance(m, nn.Linear):
            def hook(mod, inp, out, *, _m=m):
                x = inp[0]
                macs, adds = _linear_macs_adds(_m, x, out, include_bias)
                macs_by_module[name_by_module[_m]] += macs
                adds_by_module[name_by_module[_m]] += adds
            handlers.append(m.register_forward_hook(hook))

        elif isinstance(m, nn.MultiheadAttention):
            def hook(mod, inp, out, *, _m=m):
                macs, adds = _mha_macs_adds(_m, inp, out, include_bias)
                macs_by_module[name_by_module[_m]] += macs
                adds_by_module[name_by_module[_m]] += adds
            handlers.append(m.register_forward_hook(hook))

    return handlers, macs_by_module, adds_by_module, name_by_module, class_by_module

def _compute_by_component(model: nn.Module,
                          values_by_module: Dict[str, int],
                          *,
                          component_names: Optional[List[str]] = None
                          ) -> Dict[str, int]:
    if component_names is None:
        component_names = [n for n, _ in model.named_children()]
        component_set = set(component_names)
    else:
        component_set = set(component_names)

    out: Dict[str, int] = defaultdict(int)
    for mod_name, val in values_by_module.items():
        top = _top_level_component_name(mod_name)
        if top in component_set:
            out[top] += int(val)
        else:
            if component_names is None:
                out[top] += int(val)
            else:
                out["other"] += int(val)
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))

def _module_has_any_param(m: nn.Module) -> bool:
    return any(True for _ in m.parameters(recurse=False)) or any(True for _ in m.parameters())

def _module_any_param_trainable(m: nn.Module) -> bool:
    return any(p.requires_grad for p in m.parameters())

def _train_multiplier_for_module(m: nn.Module,
                                 mode: str,
                                 grad_through_frozen: bool) -> float:
    """
    Rough training estimate:
      - inference: 1×
      - train & trainable params: 3× (forward + input-grad + weight-grad)
      - train & frozen params:
          2× if gradients must pass through (forward + input-grad),
          1× if no grads pass through (forward only).
    """
    if mode == "inference":
        return 1.0
    # train mode
    if _module_has_any_param(m):
        if _module_any_param_trainable(m):
            return 3.0
        else:
            return 2.0 if grad_through_frozen else 1.0
    return 2.0 if grad_through_frozen else 1.0


def profile_flops(model: nn.Module,
                  example_inputs: Tuple[Any, ...],
                  *,
                  include_bias_add: bool = True,
                  no_grad: bool = True,
                  warmup: bool = False,
                  component_names: Optional[List[str]] = None,
                  mode: str = "inference",
                  grad_through_frozen: bool = True
                  ) -> Dict[str, Any]:
    """
    Args:
        model: nn.Module to profile.
        example_inputs: tuple of tensors passed to model(*example_inputs).
        include_bias_add: count bias adds for Conv/Linear/MHA projections.
        no_grad: run forward under torch.no_grad().
        warmup: optional dry run forward without hooks (for lazy modules).
        component_names: optional list of top-level names (e.g., ["backbone","classifier_head"])
        mode: "inference" or "train" (train multiplies per-module counts as described above).
        grad_through_frozen: whether grads must pass through frozen modules (True if you have a trainable
                             module BEFORE them, e.g., a visual prompt; False if not).

    Returns:
        dict with keys:
          - "total_macs", "total_flops"
          - "by_module_macs", "by_module_flops"
          - "by_class_macs", "by_class_flops"
          - "by_component_macs", "by_component_flops"
    """
    assert mode in {"inference", "train"}

    was_training = model.training
    model.eval()

    if warmup:
        with torch.no_grad():
            _ = model(*example_inputs)

    handlers, macs_by_module_fwd, adds_by_module_fwd, name_by_module, class_by_module = _attach_hooks(
        model, include_bias=include_bias_add
    )

    try:
        if no_grad:
            with torch.no_grad():
                _ = model(*example_inputs)
        else:
            _ = model(*example_inputs)
    finally:
        for h in handlers:
            h.remove()
        if was_training:
            model.train()

    macs_by_module = {}
    flops_by_module = {}
    total_macs = 0
    total_flops = 0

    for m, name in name_by_module.items():
        if name not in macs_by_module_fwd and name not in adds_by_module_fwd:
            continue
        fwd_macs = int(macs_by_module_fwd.get(name, 0))
        fwd_adds = int(adds_by_module_fwd.get(name, 0))

        mult = _train_multiplier_for_module(m, mode=mode, grad_through_frozen=grad_through_frozen)

        macs = int(round(fwd_macs * mult))
        flops = int(round((2 * fwd_macs + fwd_adds) * mult))

        macs_by_module[name] = macs
        flops_by_module[name] = flops

        total_macs += macs
        total_flops += flops

    by_class_macs: Dict[str, int] = defaultdict(int)
    by_class_flops: Dict[str, int] = defaultdict(int)
    for m, name in name_by_module.items():
        cls = class_by_module[m]
        by_class_macs[cls] += int(macs_by_module.get(name, 0))
        by_class_flops[cls] += int(flops_by_module.get(name, 0))

    by_component_macs = _compute_by_component(model, macs_by_module, component_names=component_names)
    by_component_flops = _compute_by_component(model, flops_by_module, component_names=component_names)

    by_module_macs_sorted = dict(sorted(macs_by_module.items(), key=lambda kv: kv[1], reverse=True))
    by_module_flops_sorted = dict(sorted(flops_by_module.items(), key=lambda kv: kv[1], reverse=True))
    by_class_macs_sorted = dict(sorted(by_class_macs.items(), key=lambda kv: kv[1], reverse=True))
    by_class_flops_sorted = dict(sorted(by_class_flops.items(), key=lambda kv: kv[1], reverse=True))

    return {
        "total_macs": int(total_macs),
        "total_flops": int(total_flops),
        "by_module_macs": by_module_macs_sorted,
        "by_module_flops": by_module_flops_sorted,
        "by_class_macs": by_class_macs_sorted,
        "by_class_flops": by_class_flops_sorted,
        "by_component_macs": by_component_macs,
        "by_component_flops": by_component_flops,
    }


def pretty_num(n: int, kind: str = "FLOPs") -> str:
    absn = float(abs(n))
    if absn >= 1e12:
        return f"{n/1e12:.3f} T{kind}"
    if absn >= 1e9:
        return f"{n/1e9:.3f} G{kind}"
    if absn >= 1e6:
        return f"{n/1e6:.3f} M{kind}"
    if absn >= 1e3:
        return f"{n/1e3:.3f} K{kind}"
    return f"{n} {kind}"

def print_report(report: Dict[str, Any], top_k: int = 15):
    print(f"Total: {pretty_num(report['total_macs'], 'MACs')} | {pretty_num(report['total_flops'], 'FLOPs')}")
    if report["by_component_flops"]:
        print("\nTop components by FLOPs:")
        for i, (name, v) in enumerate(list(report["by_component_flops"].items())[:top_k], 1):
            print(f"  {i:>2}. {name:<20} {pretty_num(v)}")
