# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""RL-MPC Environment Configuration for Deeprobotics Lite3."""

from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from rl_training.assets.deeprobotics import DEEPROBOTICS_LITE3_CFG

from .rough_env_cfg import DeeproboticsLite3RoughEnvCfg


@configclass
class RLMPCActionsCfg:
    """Action specifications for RL-MPC."""

    mpc_weights = mdp.MPCWeightsActionCfg(
        asset_name="robot",
        q_weight_bounds=(0.0, 500.0),
        r_weight_bounds=(1e-7, 1e-3),
    )


@configclass
class RLMPCObservationsCfg:
    """Observation specifications for RL-MPC."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for RL-MPC policy - outputs MPC weights."""

        # Robot state observations
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        # Previous MPC weights as input (for temporal consistency)
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for RL-MPC critic."""

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class DeeproboticsLite3RLMPCEnvCfg(DeeproboticsLite3RoughEnvCfg):
    """RL-MPC Environment Configuration for Lite3.
    
    The RL policy outputs MPC weights instead of joint positions:
    - Q weights (13-dim): [roll, pitch, yaw, x, y, z, wx, wy, wz, vx, vy, vz, gravity]
    - R weights (12-dim): Control effort weights for 4 legs x 3 forces
    
    Total action dimension: 25 (13 Q weights + 12 R weights)
    """

    def __post_init__(self):
        super().__post_init__()

        # RL-MPC specific configuration
        # Action space: MPC weights instead of joint positions
        # Q weights: 13-dim, R weights: 12-dim
        self.actions = RLMPCActionsCfg()
        
        # Update observations for RL-MPC
        self.observations = RLMPCObservationsCfg()
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        
        # Use flat terrain for initial RL-MPC training
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None
        
        # Adjust episode length for MPC training
        self.episode_length_s = 10.0
        
        # Disable zero weight rewards
        if self.__class__.__name__ == "DeeproboticsLite3RLMPCEnvCfg":
            self.disable_zero_weight_rewards()
