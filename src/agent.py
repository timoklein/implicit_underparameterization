import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_schedule(start_e: float, end_e: float, duration: float, t: int) -> float:
    """Linear annealing schedule for epsilon-greedy."""
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class NatureCNN(nn.Module):
    """CNN feature extractor from the DQN nature paper."""

    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(3136, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x / 255.0 - 0.5))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return x


class DDQN(nn.Module):
    """DDQN agent."""

    def __init__(self, env):
        super().__init__()

        self.phi = NatureCNN()
        self.q = nn.Linear(512, int(env.single_action_space.n))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning Q-values and features for calculating the regularization term."""
        train_feats = self.phi(x)
        q_vals = self.q(train_feats)
        return q_vals, train_feats
