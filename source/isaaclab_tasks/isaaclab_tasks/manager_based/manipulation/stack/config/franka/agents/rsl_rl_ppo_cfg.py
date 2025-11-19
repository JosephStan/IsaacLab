# agents/rsl_rl_ppo_cfg.py

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlPpoActorCriticCfg,
)

@configclass
class StackCubeFrankaIKRelPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "franka_stack_ik_rel"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# @configclass
# class StackCubeFrankaIKRelPPORunnerCfg(RslRlOnPolicyRunnerCfg):
#     seed = 42
#     num_steps_per_env = 24
#     max_iterations = 10000
#     save_interval = 500
#     experiment_name = "franka_stack_ik_rel"
#     device = "cuda:0"

#     algorithm = RslRlPpoAlgorithmCfg()
#     algorithm.learning_rate = 1.0e-3
#     algorithm.gamma = 0.99
#     algorithm.lam = 0.95
#     algorithm.clip_param = 0.2
#     algorithm.entropy_coef = 0.001
#     algorithm.value_loss_coef = 1.0
#     algorithm.num_learning_epochs = 5
#     algorithm.num_mini_batches = 4
#     algorithm.max_grad_norm = 1.0
#     algorithm.desired_kl = 0.01
#     algorithm.schedule = "adaptive"

#     policy = RslRlPpoActorCriticCfg()
#     policy.init_noise_std = 1.0
#     policy.actor_hidden_dims = (256, 256)
#     policy.critic_hidden_dims = (256, 256)
#     policy.activation = "elu"
