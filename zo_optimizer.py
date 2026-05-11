"""
zo_optimizer.py — Zero-order optimizer skeleton (student-implemented).

Students: Implement your gradient-free optimization logic inside
``ZeroOrderOptimizer``. The skeleton uses a 2-point central-difference
estimator as a starting point — you are expected to replace or extend it.

Key design points
-----------------
* **Layer selection** is entirely your responsibility. Set ``self.layer_names``
  to the list of parameter names you want to optimize. You can change this list
  at any time — even between ``.step()`` calls — to implement curriculum or
  progressive-layer strategies.
* **Compute budget** is enforced by ``validate.py``: ``.step()`` is called
  exactly ``n_batches`` times. Each call may invoke the model as many times as
  your estimator requires, but be mindful that more evaluations per step leave
  fewer steps in the total budget.
* **No gradients** are computed anywhere in this file. All updates must be
  derived from scalar loss values obtained by calling ``loss_fn()``.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class ZeroOrderOptimizer:
    """Gradient-free optimizer for fine-tuning a subset of model parameters.

    The optimizer maintains a list of *active* parameter names
    (``self.layer_names``). On each ``.step()`` call it perturbs only those
    parameters, estimates a pseudo-gradient from forward-pass loss values, and
    applies an update. All other parameters remain strictly frozen.

    Args:
        model:            The ``nn.Module`` to optimize.
        lr:               Step size / learning rate.   
        eps:              Perturbation magnitude for the finite-difference
                          estimator.
        perturbation_mode: Distribution used to sample the perturbation
                          direction. ``"gaussian"`` draws from N(0, I);
                          ``"uniform"`` draws from U(-1, 1) and normalises.

    Student task:
        1. Set ``self.layer_names`` to the parameter names you want to tune.
           Inspect available names with ``[n for n, _ in model.named_parameters()]``.
        2. Replace or extend ``_estimate_grad`` with a better estimator.
        3. Replace or extend ``_update_params`` with a better update rule.
        4. Optionally change ``self.layer_names`` inside ``.step()`` to
           implement dynamic layer selection strategies.

    Example — tune only the final linear layer::

        optimizer = ZeroOrderOptimizer(model)
        optimizer.layer_names = ["fc.weight", "fc.bias"]
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        eps: float = 1e-3,
        perturbation_mode: str = "bernoulli",
        beta1: float = 0.9,
        beta2: float = 0.999,
        adam_eps: float = 1e-8,
        num_directions: int = 4,
        max_update_norm: float = 0.05,
        anchor_strength: float = 0.01,
    ) -> None:
        self.model = model
        self.lr = lr
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.adam_eps = adam_eps
        self.num_directions = num_directions
        self.max_update_norm = max_update_norm
        self.anchor_strength = anchor_strength
        self.step_count = 0
        
        if perturbation_mode not in ("gaussian", "uniform","bernoulli"):
            raise ValueError(
                f"perturbation_mode must be 'gaussian' or 'uniform' or 'bernoulli', "
                f"got '{perturbation_mode}'"
            )
        self.perturbation_mode = perturbation_mode

        self.first_moment: dict[str, torch.Tensor] = {}
        self.second_moment: dict[str, torch.Tensor] = {}

        # ------------------------------------------------------------------
        # STUDENT: Set self.layer_names to the parameters you want to tune.
        #
        # The default below selects only the final classification head.
        # You may replace this with any subset of named parameters, e.g.:
        #   self.layer_names = ["layer4.1.conv2.weight", "fc.weight", "fc.bias"]
        #
        # You can also update self.layer_names inside .step() to implement
        # a dynamic schedule (e.g. gradually unfreeze deeper layers).
        # ------------------------------------------------------------------
        self.layer_names: list[str] = ["fc.bias"]
        # ------------------------------------------------------------------

        self.reference_params = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
        }

    # ------------------------------------------------------------------
    # Internal helpers — students may modify these.
    # ------------------------------------------------------------------

    def _active_params(self) -> dict[str, nn.Parameter]:
        """Return a mapping from name → parameter for all active layer names.

        Only parameters whose names appear in ``self.layer_names`` are
        returned. Parameters not in this mapping are never modified.

        Returns:
            Dict mapping parameter name to its ``nn.Parameter`` tensor.

        Raises:
            KeyError: If a name in ``self.layer_names`` does not exist in the
                      model.
        """
        named = dict(self.model.named_parameters())
        missing = [n for n in self.layer_names if n not in named]
        if missing:
            raise KeyError(
                f"The following layer names were not found in the model: "
                f"{missing}. Use [n for n, _ in model.named_parameters()] "
                f"to inspect valid names."
            )
        return {n: named[n] for n in self.layer_names}

    def _sample_direction(self, param: torch.Tensor) -> torch.Tensor:
        """Sample a random unit-norm perturbation vector of the same shape as ``param``.

        Args:
            param: The parameter tensor whose shape determines the output shape.

        Returns:
            A tensor of the same shape as ``param``, normalised to unit L2 norm.
        """
        if self.perturbation_mode == "gaussian":
            u = torch.randn_like(param)
        elif self.perturbation_mode == 'bernoulli':
            u = torch.empty_like(param).bernoulli_(0.5)
            u = u.mul_(2.0).sub_(1.0)
            return u
        else:  # uniform
            u = torch.rand_like(param) * 2.0 - 1.0

        norm = u.norm()
        if norm > 0:
            u = u / norm
        return u

    def _estimate_grad(
        self,
        loss_fn: Callable[[], float],
        params: dict[str, nn.Parameter],
    ) -> dict[str, torch.Tensor]:
        """Estimate a pseudo-gradient for each active parameter.

        Skeleton: 2-point central-difference estimator.
        For each active parameter ``p`` independently:
            1. Sample a random unit vector ``u`` of the same shape as ``p``.
            2. Evaluate  f_plus  = loss_fn() with ``p ← p + eps * u``
            3. Evaluate  f_minus = loss_fn() with ``p ← p - eps * u``
            4. Restore ``p`` to its original value.
            5. Pseudo-gradient ← ``(f_plus - f_minus) / (2 * eps) * u``

        This is an unbiased estimator of the directional derivative along ``u``
        scaled back to parameter space.

        Args:
            loss_fn: Callable that evaluates the objective on the current batch
                     and returns a scalar ``float``. May be called multiple
                     times; each call must use the *same* batch.
            params:  Dict of active parameter name → tensor (from
                     ``_active_params``).

        Returns:
            Dict mapping each parameter name to its estimated pseudo-gradient
            tensor (same shape as the parameter).

        Student task:
            Replace this with a more efficient or accurate estimator:
        """
        # ------------------------------------------------------------------
        # STUDENT: Replace or extend the gradient estimation below.
        # ------------------------------------------------------------------
        grads = {name: torch.zeros_like(param) for name, param in params.items()}

        with torch.no_grad():
            for _ in range(self.num_directions):
                directions = {
                    name: self._sample_direction(param)
                    for name, param in params.items()
                }

                for name, param in params.items():
                    param.data.add_(self.eps * directions[name])
                f_plus = loss_fn()

                for name, param in params.items():
                    param.data.sub_(2.0 * self.eps * directions[name])
                f_minus = loss_fn()

                for name, param in params.items():
                    param.data.add_(self.eps * directions[name])

                coeff = (f_plus - f_minus) / (2.0 * self.eps)
                for name in params:
                    grads[name].add_(coeff * directions[name], alpha=1.0 / self.num_directions)

        return grads
        # ------------------------------------------------------------------

    def _update_params(
        self,
        params: dict[str, nn.Parameter],
        grads: dict[str, torch.Tensor],
    ) -> None:
        """Apply the estimated pseudo-gradients to the active parameters.

        Skeleton: vanilla gradient *descent* step (minimising the loss).
            ``p ← p - lr * grad``

        Args:
            params: Dict of active parameter name → tensor.
            grads:  Dict of pseudo-gradient name → tensor (same keys as
                    ``params``).

        Student task:
            Replace with a more sophisticated update rule, e.g.:
              - Momentum: accumulate an exponential moving average of gradients.
              - Adam-style: maintain first and second moment estimates.
              - Clipped update: ``p ← p - lr * clip(grad, max_norm)``.
        """
        # ------------------------------------------------------------------
        # STUDENT: Replace or extend the parameter update below.
        # ------------------------------------------------------------------
        self.step_count += 1

        with torch.no_grad():
            for name, param in params.items():
                if name not in self.first_moment:
                    self.first_moment[name] = torch.zeros_like(param)
                    self.second_moment[name] = torch.zeros_like(param)

                grad = grads[name]
                m = self.first_moment[name]
                v = self.second_moment[name]

                m.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
                v.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)

                m_hat = m / (1.0 - self.beta1 ** self.step_count)
                v_hat = v / (1.0 - self.beta2 ** self.step_count)

                update = self.lr * m_hat / (torch.sqrt(v_hat) + self.adam_eps)
                update_norm = update.norm()
                if update_norm > self.max_update_norm:
                    update.mul_(self.max_update_norm / update_norm.clamp_min(1e-12))

                param.data.sub_(update)

                if name.endswith("weight") and self.anchor_strength > 0.0:
                    param.data.lerp_(self.reference_params[name], self.anchor_strength)
        # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, loss_fn: Callable[[], float]) -> float:
        """Perform one zero-order optimisation step.

        Calls ``loss_fn`` one or more times to estimate pseudo-gradients for
        the currently active parameters (``self.layer_names``), then applies
        an update. Parameters *not* in ``self.layer_names`` are never touched.

        Args:
            loss_fn: A callable that takes no arguments and returns a scalar
                     ``float`` representing the loss on the current mini-batch.
                     ``validate.py`` guarantees that every call to ``loss_fn``
                     within a single ``.step()`` invocation uses the *same*
                     fixed batch of data.

        Returns:
            The loss value at the *start* of the step (before any update),
            obtained from the first call to ``loss_fn()``.

        Note:
            ``validate.py`` calls ``.step()`` exactly ``n_batches`` times.
            Each forward pass inside ``loss_fn`` counts toward your compute
            budget, so prefer estimators that minimise the number of calls.
        """
        params = self._active_params()

        # Record the loss before any perturbation.
        with torch.no_grad():
            loss_before = loss_fn()

        grads = self._estimate_grad(loss_fn, params)
        self._update_params(params, grads)

        return float(loss_before)
