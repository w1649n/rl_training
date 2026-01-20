# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""MPC Controller wrapper for RL-MPC integration with Isaac Lab.

This module provides a Python wrapper to integrate the QP-MPC controller
from A1-QP-MPC-Controller repository with Isaac Lab environments.
"""

from __future__ import annotations

import torch
import numpy as np
import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Setup logger
logger = logging.getLogger(__name__)

# Try to import the MPC controller Python bindings
try:
    import mpc_controller
    MPC_AVAILABLE = True
except ImportError:
    MPC_AVAILABLE = False
    logger.warning(
        "mpc_controller module not found. Install from A1-QP-MPC-Controller repository: "
        "https://github.com/w1649n/A1-QP-MPC-Controller"
    )


class MPCControllerWrapper:
    """Wrapper class for the QP-MPC controller.
    
    This class manages multiple MPC instances for parallel environments
    and provides batched state input/GRF output interface.
    
    Robot Configuration (Lite3):
        - Mass: ~15.0 kg
        - Foot order: FL, FR, RL, RR (matching MPC convention)
        - Body frame: x-forward, y-left, z-up
    """

    def __init__(
        self,
        num_envs: int,
        device: str = "cuda:0",
        robot_mass: float = 15.0,
        dt: float = 0.0025,
    ):
        """Initialize MPC controller wrapper.
        
        Args:
            num_envs: Number of parallel environments
            device: PyTorch device for tensors
            robot_mass: Robot mass in kg
            dt: Control timestep in seconds
        """
        self.num_envs = num_envs
        self.device = device
        self.robot_mass = robot_mass
        self.dt = dt
        
        if not MPC_AVAILABLE:
            raise RuntimeError(
                "MPC controller not available. Please install from "
                "https://github.com/w1649n/A1-QP-MPC-Controller"
            )
        
        # Create MPC instances for each environment
        self.mpc_instances = [mpc_controller.ConvexMpc() for _ in range(num_envs)]
        
        # Output buffer for ground reaction forces
        self.grf = torch.zeros(num_envs, 12, device=device)

    def set_weights(
        self,
        q_weights: torch.Tensor,
        r_weights: torch.Tensor,
    ):
        """Set MPC weights for all environments.
        
        Args:
            q_weights: State cost weights (num_envs, 13)
            r_weights: Control cost weights (num_envs, 12)
        """
        q_np = q_weights.cpu().numpy()
        r_np = r_weights.cpu().numpy()
        
        for i, mpc in enumerate(self.mpc_instances):
            mpc.set_weights(q_np[i], r_np[i])

    def solve(
        self,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        foot_pos: torch.Tensor,
        contacts: torch.Tensor,
        desired_vel: torch.Tensor,
        desired_yaw_rate: torch.Tensor,
    ) -> torch.Tensor:
        """Solve MPC for all environments.
        
        Args:
            root_pos: Base position (num_envs, 3)
            root_quat: Base quaternion wxyz (num_envs, 4)
            root_lin_vel: Base linear velocity in body frame (num_envs, 3)
            root_ang_vel: Base angular velocity in body frame (num_envs, 3)
            foot_pos: Foot positions in world frame (num_envs, 4, 3)
            contacts: Contact states (num_envs, 4) boolean
            desired_vel: Desired linear velocity (num_envs, 3)
            desired_yaw_rate: Desired yaw rate (num_envs, 1)
            
        Returns:
            Ground reaction forces (num_envs, 12) in body frame
        
        Note:
            The MPC solving is performed sequentially for each environment.
            This is a limitation of the current MPC library implementation.
            Future optimization could parallelize this if the library supports it.
        """
        # Convert to numpy for MPC solver
        root_pos_np = root_pos.cpu().numpy()
        root_quat_np = root_quat.cpu().numpy()
        root_lin_vel_np = root_lin_vel.cpu().numpy()
        root_ang_vel_np = root_ang_vel.cpu().numpy()
        foot_pos_np = foot_pos.cpu().numpy()
        contacts_np = contacts.cpu().numpy()
        desired_vel_np = desired_vel.cpu().numpy()
        desired_yaw_rate_np = desired_yaw_rate.cpu().numpy()
        
        grf_results = []
        
        for i in range(self.num_envs):
            # Create MPC state
            state = mpc_controller.MPCState()
            
            # Convert quaternion to euler angles
            state.root_euler = self._quat_to_euler(root_quat_np[i])
            state.root_pos = root_pos_np[i]
            state.root_lin_vel = root_lin_vel_np[i]
            state.root_ang_vel = root_ang_vel_np[i]
            
            # Foot positions (3x4 matrix)
            state.foot_pos = foot_pos_np[i].T  # (4, 3) -> (3, 4)
            
            # Contact states
            state.contacts = contacts_np[i].tolist()
            
            # Robot parameters
            state.robot_mass = self.robot_mass
            
            # Desired trajectory
            state.root_pos_d = np.array([0.0, 0.0, 0.35])  # Target height
            state.root_euler_d = np.array([0.0, 0.0, state.root_euler[2]])  # Keep current yaw
            state.root_lin_vel_d = desired_vel_np[i]
            state.root_ang_vel_d = np.array([0.0, 0.0, desired_yaw_rate_np[i, 0]])
            
            # Solve MPC
            grf = self.mpc_instances[i].solve(state, dt=self.dt)
            grf_results.append(grf)
        
        # Convert back to torch tensor
        self.grf = torch.tensor(
            np.array(grf_results), 
            dtype=torch.float32, 
            device=self.device
        )
        
        return self.grf

    def _quat_to_euler(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion (wxyz) to euler angles (roll, pitch, yaw)."""
        w, x, y, z = quat
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])

    def grf_to_joint_torques(
        self,
        grf: torch.Tensor,
        jacobians: torch.Tensor,
    ) -> torch.Tensor:
        """Convert ground reaction forces to joint torques using Jacobian transpose.
        
        Args:
            grf: Ground reaction forces (num_envs, 12) [FL_xyz, FR_xyz, RL_xyz, RR_xyz]
            jacobians: Foot Jacobians (num_envs, 4, 3, 3) for each leg
            
        Returns:
            Joint torques (num_envs, 12)
        """
        # Reshape GRF to (num_envs, 4, 3)
        grf_per_leg = grf.view(self.num_envs, 4, 3)
        
        # τ = J^T @ F for each leg
        # jacobians: (num_envs, 4, 3, 3) - 3 joints x 3 force directions
        # grf_per_leg: (num_envs, 4, 3) - 3 force directions
        
        torques = torch.zeros(self.num_envs, 12, device=self.device)
        
        for leg_idx in range(4):
            J_T = jacobians[:, leg_idx].transpose(-2, -1)  # (num_envs, 3, 3)
            F = grf_per_leg[:, leg_idx].unsqueeze(-1)  # (num_envs, 3, 1)
            tau = torch.bmm(J_T, F).squeeze(-1)  # (num_envs, 3)
            torques[:, leg_idx*3:(leg_idx+1)*3] = tau
        
        return torques

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """Reset MPC instances for specified environments."""
        if env_ids is None:
            for mpc in self.mpc_instances:
                mpc.reset()
        else:
            for i in env_ids.cpu().numpy():
                self.mpc_instances[i].reset()
