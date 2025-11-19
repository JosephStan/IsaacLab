# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_rl.rsl_rl import RslRlDistillationRunnerCfg
from isaaclab_rl.rsl_rl.distillation_cfg import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationStudentTeacherCfg,
)
from isaaclab.utils import configclass

@configclass
class StackCubeFrankaIKRelDistilRunnerCfg(RslRlDistillationRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = "franka_stack_ik_rel_distil"
    device = "cuda:0"

    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="Distillation",
        num_learning_epochs=5,
        learning_rate=1e-3,
        gradient_length=24,
        max_grad_norm=1.0,
        imitation_dataset_path="~/IsaacLab/datasets/generated_dataset.hdf5",
    )
    policy = RslRlDistillationStudentTeacherCfg(
        class_name="StudentTeacher",
        init_noise_std=1.0,
        noise_std_type="scalar",
        student_hidden_dims=[256, 256],
        teacher_hidden_dims=[256, 256],
        activation="elu",
    )
