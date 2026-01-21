# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""RSL-RL PPO configuration for RL-MPC training."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class DeeproboticsLite3RLMPCPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO Runner configuration for RL-MPC training.
    
    Key differences from standard locomotion training:
    - Action space: 25-dim (13 Q weights + 12 R weights) instead of 12-dim joint positions
    - Larger network for learning weight-to-performance mapping
    - Lower learning rate for stable MPC weight learning
    """
    
    num_steps_per_env = 24
    max_iterations = 20000  # More iterations for RL-MPC
    save_interval = 200
    experiment_name = "deeprobotics_lite3_rl_mpc"
    empirical_normalization = False
    clip_actions = 1.0  # Actions are normalized [-1, 1]
    
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,  # Lower initial noise for MPC weights
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 256, 128],  # Deeper network
        critic_hidden_dims=[512, 256, 256, 128],
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,  # Lower entropy for more deterministic weights
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,  # Lower learning rate
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
