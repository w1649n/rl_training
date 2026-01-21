# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""MPC-based action configuration for RL-MPC training."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class MPCWeightsAction(ActionTerm):
    """Action term that outputs MPC weights for RL-MPC integration.
    
    The action space consists of:
    - Q weights (13-dim): State cost weights [roll, pitch, yaw, x, y, z, wx, wy, wz, vx, vy, vz, gravity]
    - R weights (12-dim): Control cost weights for ground reaction forces
    
    These weights are used by the external MPC controller to compute optimal GRF.
    """

    cfg: MPCWeightsActionCfg
    
    def __init__(self, cfg: MPCWeightsActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        # Action dimensions
        self.q_weights_dim = 13
        self.r_weights_dim = 12
        self._action_dim = self.q_weights_dim + self.r_weights_dim
        
        # Default MPC weights (from A1-QP-MPC-Controller)
        self.default_q_weights = torch.tensor([
            80.0, 80.0, 1.0,      # roll, pitch, yaw weights
            0.0, 0.0, 270.0,      # x, y, z position weights
            1.0, 1.0, 20.0,       # angular velocity weights
            20.0, 20.0, 20.0,     # linear velocity weights
            0.0                   # gravity placeholder
        ], device=self._device)
        
        self.default_r_weights = torch.tensor([
            1e-5, 1e-5, 1e-6,     # FL leg forces
            1e-5, 1e-5, 1e-6,     # FR leg forces
            1e-5, 1e-5, 1e-6,     # RL leg forces
            1e-5, 1e-5, 1e-6      # RR leg forces
        ], device=self._device)
        
        # Initialize processed actions
        self._processed_actions = torch.zeros(
            self.num_envs, self._action_dim, device=self._device
        )
        
        # Weight bounds for safety
        self.q_weight_bounds = cfg.q_weight_bounds
        self.r_weight_bounds = cfg.r_weight_bounds
        
        # Validate R weight bounds (must be positive for log scale)
        if cfg.r_weight_bounds[0] <= 0 or cfg.r_weight_bounds[1] <= 0:
            raise ValueError(
                f"R weight bounds must be positive for logarithmic scaling. "
                f"Got: {cfg.r_weight_bounds}"
            )
        
        # Precompute scaling constants for R weights (log scale)
        # R weights use log scale because they span multiple orders of magnitude (1e-7 to 1e-3)
        # Formula: r = exp((a+1)/2 * (log(r_max) - log(r_min)) + log(r_min))
        # where a is the action in [-1, 1]
        self.log_r_min = torch.tensor(cfg.r_weight_bounds[0], device=self._device).log()
        self.log_r_max = torch.tensor(cfg.r_weight_bounds[1], device=self._device).log()
        self.log_r_range = self.log_r_max - self.log_r_min

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """Process raw actions to MPC weights.
        
        The network outputs normalized actions in [-1, 1], which are scaled
        to appropriate MPC weight ranges. Q weights use linear scaling while
        R weights use logarithmic scaling due to their wide range.
        """
        self._raw_actions = actions.clone()
        
        # Split actions into Q and R weights
        q_actions = actions[:, :self.q_weights_dim]
        r_actions = actions[:, self.q_weights_dim:]
        
        # Scale Q weights: linear mapping from [-1, 1] to [q_min, q_max]
        q_min, q_max = self.q_weight_bounds
        q_weights = (q_actions + 1.0) / 2.0 * (q_max - q_min) + q_min
        
        # Scale R weights: logarithmic mapping from [-1, 1] to [r_min, r_max]
        # Using precomputed log constants for efficiency
        r_weights = torch.exp(
            (r_actions + 1.0) / 2.0 * self.log_r_range + self.log_r_min
        )
        
        self._processed_actions = torch.cat([q_weights, r_weights], dim=-1)

    def apply_actions(self):
        """Apply MPC weights - no direct physics application.
        
        The MPC weights are stored in processed_actions and should be
        retrieved by the MPC controller wrapper.
        """
        pass

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset to default MPC weights."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._device)
        
        default_actions = torch.cat([
            self.default_q_weights.unsqueeze(0).expand(len(env_ids), -1),
            self.default_r_weights.unsqueeze(0).expand(len(env_ids), -1)
        ], dim=-1)
        
        self._processed_actions[env_ids] = default_actions

    def get_q_weights(self) -> torch.Tensor:
        """Get current Q weights for all environments."""
        return self._processed_actions[:, :self.q_weights_dim]

    def get_r_weights(self) -> torch.Tensor:
        """Get current R weights for all environments."""
        return self._processed_actions[:, self.q_weights_dim:]


@configclass
class MPCWeightsActionCfg(ActionTermCfg):
    """Configuration for MPC weights action term."""

    class_type: type = MPCWeightsAction

    # Weight bounds for Q (state cost)
    q_weight_bounds: tuple[float, float] = (0.0, 500.0)
    
    # Weight bounds for R (control cost) - log scale
    r_weight_bounds: tuple[float, float] = (1e-7, 1e-3)
