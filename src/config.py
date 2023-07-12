from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration for a feature regularized DQN agent."""

    # Experiment settings
    exp_name: str = "Feature-Regularized-DQN"
    seed: int = 0
    torch_deterministic: bool = False
    gpu: Optional[int] = 1
    track: bool = False
    wandb_project_name: str = "Atari_feature_regularizers"
    wandb_entity: Optional[str] = None
    capture_video: bool = False
    save_model: bool = False

    # Environment settings
    env_id: str = "PongNoFrameskip-v4"
    total_timesteps: int = 6000000
    num_envs: int = 1

    # DQN settings
    buffer_size: int = 1000000
    learning_rate: float = 1e-4
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 1000
    batch_size: int = 32
    start_e: float = 1.0
    end_e: float = 0.01
    exploration_fraction: float = 0.10
    learning_starts: int = 80000  # default: 80000
    train_frequency: int = 4

    # IU-DQN settings
    regularize: bool = False
    regularization_coefficient: float = 0.0005
    srank_threshold: float = 0.99
