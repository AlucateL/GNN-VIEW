# Standard library
from typing import Union

# Third-party
import torch

# First-party
from neural_lam.models.EDM import EDM

# Local
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore


class FM(EDM):
    """
    Flow Matching Forecasting Model.
    """

    def __init__(
        self,
        args,
        config: NeuralLAMConfig,
        datastore: BaseDatastore,
        datastore_boundary: Union[BaseDatastore, None],
        datastore_atmosphere: Union[BaseDatastore, None],
        **kwargs,
    ):
        super().__init__(
            args,
            config,
            datastore,
            datastore_boundary,
            datastore_atmosphere,
            **kwargs,
        )

    # Evaluation
    def predict_step(
        self,
        prev_state,
        prev_prev_state,
        forcing,
        boundary_forcing,
        atmosphere_forcing,
    ):
        """
        Step state one step ahead using prediction model, X_{t-1}, X_t -> X_t+1
        prev_state: (B, num_grid_nodes, feature_dim), X_t
        prev_prev_state: (B, num_grid_nodes, feature_dim), X_{t-1}
        forcing: (B, num_grid_nodes, forcing_dim)
        boundary_forcing: (B, num_boundary_nodes, boundary_forcing_dim)
        atmosphere_forcing: (B, num_atmosphere_nodes, atmosphere_forcing_dim)


        Returns:
        next_state: (B, N_grid, d_state),
            predicted weather state X_{t+1} at time t+1
        """
        if prev_prev_state is not None:
            input_grid = torch.cat(
                (prev_state, prev_prev_state, forcing), dim=-1
            )
        else:
            input_grid = torch.cat((prev_state, forcing), dim=-1)
        # (B, N_grid, d_input)

        latents = torch.randn_like(prev_state)  # (B, N_grid, d_state)

        # Run through sampler
        if self.sampler == "heun":
            next_state = self.heun_sampler(
                latents=latents,
                class_labels=input_grid,
                boundary_forcing=boundary_forcing,
                atmosphere_forcing=atmosphere_forcing,
            )
        elif self.sampler == "stochastic":
            next_state = self.stochastic_sampler(
                latents=latents,
                class_labels=input_grid,
                boundary_forcing=boundary_forcing,
                atmosphere_forcing=atmosphere_forcing,
            )

        # Add residual if needed
        if self.pred_residual:
            next_state = (
                next_state * self.diff_std
            ) + self.diff_mean  # Unormalize residual
            next_state = prev_state + next_state

        return next_state

    def predict_step_train(
        self,
        prev_state,
        prev_prev_state,
        forcing,
        boundary_forcing,
        atmosphere_forcing,
        target_state,
    ):
        """
        Predict weather state one time step ahead
        X_{t-1}, X_t -> X_t+1

        prev_state: (B, N_grid, d_state), weather state X_t at time t
        prev_prev_state: (B, N_grid, d_state), weather state X_{t-1} at time t-1
        batch_static_features: (B, N_grid, batch_static_feature_dim), static
        forcing: (B, N_grid, forcing_dim), dynamic forcing
        boundary_forcing: (B, num_boundary_nodes, boundary_forcing_dim)
        atmosphere_forcing: (B, num_atmosphere_nodes, atmosphere_forcing_dim)
        target_state: (B, N_grid, d_state), true state X_{t+1} at time t+1

        Returns:
        next_state: (B, N_grid, d_state), predicted weather state X_{t+1} at t+1
        loss: (B)
        """
        if prev_prev_state is not None:
            input_grid = torch.cat(
                (prev_state, prev_prev_state, forcing), dim=-1
            )
        else:
            input_grid = torch.cat((prev_state, forcing), dim=-1)

        # Make y residual if needed
        if self.pred_residual:
            y = target_state - prev_state
            y = (y - self.diff_mean) / self.diff_std  # Normalize residual
        else:
            y = target_state

        z0 = torch.randn_like(y)
        z1 = y
        t = torch.rand([prev_state.shape[0], 1, 1], device=prev_state.device)
        zt = (1 - t) * z0 + t * z1

        # Shape (B, d_state, N_x, N_y)
        pred_drift = self.model(
            zt, t, input_grid, boundary_forcing, atmosphere_forcing
        )

        # This predicts the drift b
        drift = z1 - z0

        pred_std = self.per_var_std

        loss = self.loss(
            pred_drift,
            drift,
            pred_std,
            mask=self.loss_mask,
        )  # (B)

        # This is the predicted E[z1 | zt]
        next_state = zt + pred_drift * (1 - t)

        # Add residual if needed
        if self.pred_residual:
            next_state = (
                next_state * self.diff_std
            ) + self.diff_mean  # Unormalize residual
            next_state = prev_state + next_state

        return next_state, loss

    # ----------------------------------------------------------------------------
    # Sampling methods below adapted from:
    # Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    #
    # This work is licensed under a Creative Commons
    # Attribution-NonCommercial-ShareAlike 4.0 International License.
    # You should have received a copy of the license along with this
    # work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models"."""

    # ----------------------------------------------------------------------------
    # Proposed Heun sampler (Algorithm 1).
    def heun_sampler(
        self,
        latents,
        class_labels=None,
        boundary_forcing=None,
        atmosphere_forcing=None,
        num_steps=20,
    ):
        tmin = 0.0
        tmax = 1.0

        # Time step discretization.
        t_steps = torch.linspace(tmin, tmax, num_steps, device=self.device)

        # Main sampling loop.
        x_next = latents
        # 0, ..., N-1
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            t_cur = t_cur.reshape(-1, 1, 1).flatten()
            t_next = t_next.reshape(-1, 1, 1).flatten()

            # Euler step.
            x_cur = x_next
            d_cur = self.model(
                x_cur, t_cur, class_labels, boundary_forcing, atmosphere_forcing
            )
            x_next = x_cur + (t_next - t_cur) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                d_prime = self.model(
                    x_next,
                    t_next,
                    class_labels,
                    boundary_forcing,
                    atmosphere_forcing,
                )
                x_next = x_cur + (t_next - t_cur) * (
                    0.5 * d_cur + 0.5 * d_prime
                )

        return x_next

    def stochastic_sampler(
        self,
        latents,
        class_labels=None,
        boundary_forcing=None,
        atmosphere_forcing=None,
        num_steps=20,
    ):
        tmin = 0.0
        tmax = 1
        eps = 1.0

        # Time step discretization.
        ts = torch.linspace(tmin, tmax, num_steps, device=self.device)
        dt = (tmax - tmin) / num_steps

        # Main sampling loop.
        zt = latents  # Initialize with noise
        for t in ts[:-1]:
            alpha_t = t
            beta_t = 1 - t
            gamma_t = 1
            alpha_dot_t = 1
            # beta_dot_t = -1
            eps_t = eps * beta_t

            b = self.model(
                zt, t, class_labels, boundary_forcing, atmosphere_forcing
            )
            s = (alpha_t * b - alpha_dot_t * zt) / (
                beta_t * gamma_t
            )  # s = (t * b - zt) / (1 - t)
            dz = b + eps_t * s

            dW = torch.randn_like(zt) * torch.sqrt(2 * dt * eps_t)

            zt = zt + dz * dt + dW

        return zt  # (B, N_grid, d_state)
