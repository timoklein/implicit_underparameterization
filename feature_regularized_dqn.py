"""
Implementation of feature regularized DDQN based on cleanRL's DQN Atari implementation:
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py

Minor modification compared to cleanRL's base agent is the usage of double Q-Learning.
"""

import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pyrallis
import torch
import torch._dynamo as dynamo
import torch.nn.functional as F
import torch.optim as optim

import wandb
from src.agent import DDQN, linear_schedule
from src.buffer import ReplayBuffer, ReplayBufferSamples
from src.config import Config
from src.feat_rank import effective_rank
from src.utils import make_env, set_cuda_configuration


def update(
    q_network: DDQN, target_network: DDQN, optimizer: optim.Adam, data: ReplayBufferSamples, cfg: Config
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate DDQN and feature regularization losses and take a gradient step."""

    with torch.no_grad():
        # Get value estimates from the target network
        target_vals, _ = target_network.forward(data.next_observations)
        # Select actions through the policy network
        policy_actions = q_network(data.next_observations)[0].argmax(dim=1)
        target_max = target_vals[range(len(target_vals)), policy_actions]
        # Calculate Q-target
        td_target = data.rewards.flatten() + cfg.gamma * target_max * (1 - data.dones.flatten())

    qs, train_feats = q_network(data.observations)
    old_val = qs.gather(1, data.actions).squeeze()

    # Calculate loss
    dqn_loss = F.mse_loss(td_target, old_val)
    if cfg.regularize:
        # Get the singular values of the feature matrix and calculate the regularization loss
        _, s_min, s_max = effective_rank(train_feats, cfg.srank_threshold)
        # Catch case where the rank of the feature matrix has collapsed
        regularizer = cfg.regularization_coefficient * (s_max**2 - s_min**2)
        loss = dqn_loss + regularizer
    else:
        loss = dqn_loss
        # Placeholder for the regularization loss
        regularizer = torch.tensor(0.0, requires_grad=False)

    # optimize the model
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(q_network.parameters(), 1e10, norm_type=2.0)
    optimizer.step()

    return loss, dqn_loss, regularizer, old_val, grad_norm


def main(cfg: Config) -> None:
    """Main training method for feature regularized DDQN."""
    run_name = f"{cfg.env_id}__{cfg.exp_name}__{cfg.seed}__{int(time.time())}"

    wandb.init(
        project=cfg.wandb_project_name,
        entity=cfg.wandb_entity,
        config=vars(cfg),
        name=run_name,
        monitor_gym=True,
        save_code=False,
        mode="online" if cfg.track else "disabled",
    )

    if cfg.save_model:
        evaluation_episode = 0
        wandb.define_metric("evaluation_episode")
        wandb.define_metric("eval/episodic_return", step_metric="evaluation_episode")

    # TRY NOT TO MODIFY: seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.use_deterministic_algorithms(cfg.torch_deterministic)

    device = set_cuda_configuration(cfg.gpu)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(cfg.env_id, cfg.seed + i, i, cfg.capture_video, run_name) for i in range(cfg.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = DDQN(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate)
    target_network = DDQN(envs).to(device)
    target_network.load_state_dict(agent.state_dict())

    rb = ReplayBuffer(
        cfg.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=cfg.seed)
    for global_step in range(cfg.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(cfg.start_e, cfg.end_e, cfg.exploration_fraction * cfg.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values, _ = agent(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if "episode" not in info:
                    continue
                epi_return = info["episode"]["r"].item()
                print(f"global_step={global_step}, episodic_return={epi_return}")
                wandb.log(
                    {
                        "charts/episodic_return": epi_return,
                        "charts/episodic_length": info["episode"]["l"].item(),
                        "charts/epsilon": epsilon,
                    },
                    step=global_step,
                )

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncated):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminated, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > cfg.learning_starts:
            if global_step % cfg.train_frequency == 0:
                data = rb.sample(cfg.batch_size)

                # Check for graph breaks in torch.compile
                # breaks = dynamo.explain(
                #     update, agent, target_network, optimizer, data, cfg.gamma, cfg.regularize, cfg.regularization_coefficient, cfg.srank_threshold
                # )[-1]

                loss, dqn_loss, regularizer, old_val, grad_norm = update(
                    q_network=agent, target_network=target_network, optimizer=optimizer, data=data, cfg=cfg
                )

                if global_step % 100 == 0:
                    # Calculate the effective rank for logging
                    with torch.inference_mode():
                        data = rb.sample(5000)
                        feats = agent.phi(data.observations)
                        srank, s_min, s_max = effective_rank(feats, cfg.srank_threshold)

                    print("SPS:", int(global_step / (time.time() - start_time)))
                    wandb.log(
                        {
                            "losses/loss": loss.item(),
                            "losses/td_loss": dqn_loss.item(),
                            "losses/q_values": old_val.mean().item(),
                            "losses/grad_norm": grad_norm.item(),
                            "regularization/regularization_loss": regularizer.item(),
                            "regularization/s_min": s_min.item(),
                            "regularization/s_max": s_max.item(),
                            "charts/srank": srank.item() + 1,  # offset indices starting with 0
                            "charts/SPS": int(global_step / (time.time() - start_time)),
                        },
                        step=global_step,
                    )

            # update target network
            if global_step % cfg.target_network_frequency == 0:
                agent_params = list(agent.phi.parameters()) + list(agent.q.parameters())
                target_params = list(target_network.phi.parameters()) + list(target_network.q.parameters())
                for target_network_param, q_network_param in zip(target_params, agent_params):
                    target_network_param.data.copy_(
                        cfg.tau * q_network_param.data + (1.0 - cfg.tau) * target_network_param.data
                    )

    if cfg.save_model:
        model_path = Path(f"runs/{run_name}/{cfg.exp_name}")
        model_path.mkdir(parents=True, exist_ok=True)
        torch.save(agent.state_dict(), model_path / ".cleanrl_model")
        print(f"model saved to {model_path}")
        from src.evaluate import evaluate

        episodic_returns = evaluate(
            model_path=model_path,
            make_env=make_env,
            env_id=cfg.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=agent,
            device=device,
            epsilon=0.05,
            capture_video=False,
        )
        for episodic_return in episodic_returns:
            wandb.log({"evaluation_episode": evaluation_episode, "eval/episodic_return": episodic_return})
            evaluation_episode += 1

    envs.close()
    wandb.finish()


if __name__ == "__main__":
    cfg = pyrallis.parse(config_class=Config)
    main(cfg)
