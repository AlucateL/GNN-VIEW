"""Single-optimizer wrapper around the official ``torch.optim.Muon`` + AdamW.

``torch.optim.Muon`` (PyTorch >= 2.11) optimizes only 2D hidden weight
matrices and *raises* on any parameter with ``ndim != 2``; embeddings, the
output layer and all 1D parameters (biases, norm gains) must be optimized by a
standard method such as AdamW. It also has no built-in aux-Adam group, so Muon
must be paired with a separate ``torch.optim.AdamW``.

PyTorch Lightning's automatic optimization (with ``gradient_clip_val`` /
``accumulate_grad_batches``) supports only a *single* optimizer object. This
class therefore presents one ``torch.optim.Optimizer`` to Lightning while
delegating the actual update to the two official optimizers. It contains no
numerical optimization logic of its own.

Conv (4D) weights can optionally be optimized by Muon by flattening them to 2D:
a contiguous weight ``p`` of shape ``[out, in, kH, kW]`` has a storage-sharing
2D view ``v = p.view(out, -1)``. Since ``torch.optim.Muon`` refuses non-leaf
tensors, we detach the view and mark it as a leaf (it still shares storage with
``p``). Each step we point ``v.grad`` at the real (already grad-clipped)
gradient and let Muon update ``v`` in place, which writes straight back to ``p``.
"""

# Third-party
import torch

# First-party
# ``torch.optim.Muon`` only exists in PyTorch >= 2.9; the current environment
# ships 2.5.1, so we use a vendored copy of the official implementation.
from ._muon import Muon


class MuonAuxAdam(torch.optim.Optimizer):
    """Delegating wrapper: Muon for 2D (+ flattened conv) weights, AdamW rest.

    Parameters
    ----------
    muon_params : list[torch.nn.Parameter]
        2D weight matrices optimized directly by ``torch.optim.Muon``.
    conv_muon_params : list[torch.nn.Parameter]
        Contiguous >2D (e.g. 4D conv) weights to be flattened to 2D and
        optimized by Muon. Empty for the plain ``muon`` option.
    adamw_params : list[torch.nn.Parameter]
        Everything else (1D params, embeddings, output layer) optimized by
        ``torch.optim.AdamW``.
    lr, weight_decay, momentum, betas
        Shared hyperparameters. ``lr``/``weight_decay`` transfer directly from
        AdamW tuning thanks to ``adjust_lr_fn="match_rms_adamw"``.
    """

    def __init__(
        self,
        muon_params,
        conv_muon_params,
        adamw_params,
        lr,
        weight_decay=0.0,
        momentum=0.95,
        betas=(0.9, 0.95),
    ):
        muon_params = list(muon_params)
        conv_muon_params = list(conv_muon_params)
        adamw_params = list(adamw_params)

        # Flatten each contiguous conv weight to a storage-sharing 2D *leaf*
        # view (Muon rejects non-leaf tensors). (p_real, v_flat) pairs are kept
        # so we can route gradients into the views each step.
        self._conv_pairs = []
        for p in conv_muon_params:
            assert p.is_contiguous(), (
                "MuonAuxAdam expects contiguous conv weights; "
                "non-contiguous params should be routed to AdamW instead."
            )
            v = p.detach().view(p.shape[0], -1)
            v.requires_grad_(True)
            self._conv_pairs.append((p, v))

        muon_all = muon_params + [v for _, v in self._conv_pairs]

        self._muon = (
            Muon(
                muon_all,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                adjust_lr_fn="match_rms_adamw",
            )
            if muon_all
            else None
        )
        self._adamw = (
            torch.optim.AdamW(
                adamw_params, lr=lr, betas=betas, weight_decay=weight_decay
            )
            if adamw_params
            else None
        )
        assert self._muon is not None or self._adamw is not None, (
            "MuonAuxAdam received no parameters to optimize."
        )

        # Register the *real* parameters with the base Optimizer so that the LR
        # scheduler and Lightning's gradient clipping operate on the true
        # gradients (including the 4D conv grads). The base optimizer never
        # steps these itself; step() delegates to the two sub-optimizers.
        real_params = (
            muon_params
            + [p for p, _ in self._conv_pairs]
            + adamw_params
        )
        super().__init__(real_params, {"lr": lr})

    @torch.no_grad()
    def step(self, closure=None):
        # In Lightning automatic optimization the closure runs the backward
        # pass and applies gradient clipping (on the real params above) before
        # returning. Re-enable grad for it (step() itself is no_grad).
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Route the (already-clipped) conv gradients into the 2D views Muon
        # optimizes. reshape() is safe here: Muon only reads the gradient.
        for p, v in self._conv_pairs:
            v.grad = None if p.grad is None else p.grad.reshape(p.shape[0], -1)

        # Propagate the scheduled LR to the sub-optimizers. Both groups share a
        # single cosine schedule, so param_groups[0]["lr"] is authoritative.
        lr = self.param_groups[0]["lr"]
        for opt in (self._muon, self._adamw):
            if opt is not None:
                for group in opt.param_groups:
                    group["lr"] = lr

        if self._muon is not None:
            self._muon.step()
        if self._adamw is not None:
            self._adamw.step()
        return loss

    def zero_grad(self, set_to_none=True):
        # Zero the real gradients via the base optimizer; also drop the
        # transient view gradients.
        super().zero_grad(set_to_none=set_to_none)
        for _, v in self._conv_pairs:
            v.grad = None

    def state_dict(self):
        return {
            "muon": None if self._muon is None else self._muon.state_dict(),
            "adamw": None if self._adamw is None else self._adamw.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if self._muon is not None and state_dict.get("muon") is not None:
            self._muon.load_state_dict(state_dict["muon"])
        if self._adamw is not None and state_dict.get("adamw") is not None:
            self._adamw.load_state_dict(state_dict["adamw"])
