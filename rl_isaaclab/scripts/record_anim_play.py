"""Record a USD animation from a Sharpa PPO or ProprioAdapt checkpoint."""

import argparse
from typing import cast

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Record a Sharpa policy rollout as USD animation.")
parser.add_argument("--task", type=str, required=True, help="Registered Isaac Lab task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=42, help="Environment seed.")
parser.add_argument("--cache", type=str, default=None, help="Grasp cache path.")
parser.add_argument("--load_path", type=str, required=True, help="Checkpoint path.")
parser.add_argument("--algorithm", type=str, default=None, help="Policy class override.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Isaac Lab blocks animation recording in headless mode. Kit can still run
# windowless on a server without a display when --headless is omitted.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

import rl_isaaclab.tasks.inhand_rotate  # noqa: E402,F401
from rl_isaaclab.algo.padapt.padapt import ProprioAdapt  # noqa: E402
from rl_isaaclab.algo.ppo.ppo import PPO  # noqa: E402
from rl_isaaclab.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_isaaclab.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402


def main():
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "agent_cfg_entry_point")

    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg["seed"] = args_cli.seed
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg["device"] = args_cli.device if args_cli.device is not None else agent_cfg["device"]
    agent_cfg["algo"] = args_cli.algorithm if args_cli.algorithm is not None else agent_cfg["algo"]
    agent_cfg["load_path"] = args_cli.load_path
    env_cfg.reset_random_quat = False
    env_cfg.randomize_pd_gains = False
    env_cfg.randomize_friction = True
    env_cfg.randomize_com = False
    env_cfg.randomize_mass = False
    env_cfg.randomize_joint_pos_offset = False
    env_cfg.sim.gravity = (0, 0, -9.81)
    env_cfg.gravity_curriculum = False
    env_cfg.grasp_cache_path = args_cli.cache if args_cli.cache is not None else env_cfg.grasp_cache_path

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    agent = eval(agent_cfg["algo"])(
        env,
        output_dir=None,
        full_config=ConfigWrapper(agent_cfg, env_cfg, test=True),
        create_output_dir=False,
    )
    agent.restore_test(agent_cfg["load_path"])
    agent.set_eval()
    obs_dict = cast(dict[str, torch.Tensor], env.reset())

    while simulation_app.is_running():
        with torch.inference_mode():
            if isinstance(agent, PPO):
                input_dict = {
                    "obs": agent.running_mean_std(obs_dict["obs"]),
                    "priv_info": obs_dict["priv_info"],
                }
            elif isinstance(agent, ProprioAdapt):
                input_dict = {
                    "obs": agent.running_mean_std(obs_dict["obs"]),
                    "proprio_hist": agent.sa_mean_std(obs_dict["proprio_hist"].detach()),
                }
            else:
                raise ValueError(f"Unsupported policy type: {type(agent).__name__}")
            actions = torch.clamp(agent.model.act_inference(input_dict), -1.0, 1.0)
            obs_dict, _, _, _ = env.step(actions)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()