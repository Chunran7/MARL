#!/usr/bin/env python
import sys
import os
curPath = os.path.abspath(__file__)

if len(curPath.split('/'))==1:
    rootPath = '\\'.join(curPath.split('\\')[:-3])
else:
    rootPath = '/'.join(curPath.split('/')[:-3])
sys.path.append(os.path.split(rootPath)[0])

import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from mappo_lagrangian.config import get_config, AsyncConfig, HIERARCHICAL_CONFIG, IMPORTANCE_CONFIG, PROGRESSIVE_CONFIG
from mappo_lagrangian.envs.safety_ma_mujoco.safety_multiagent_mujoco import MujocoMulti
from mappo_lagrangian.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "mujoco":
                env_args = {"scenario": all_args.scenario,
                            "agent_conf": all_args.agent_conf,
                            "agent_obsk": all_args.agent_obsk,
                            "episode_limit": 1000}
                env = MujocoMulti(env_args=env_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "mujoco":
                env_args = {"scenario": all_args.scenario,
                            "agent_conf": all_args.agent_conf,
                            "agent_obsk": all_args.agent_obsk,
                            "episode_limit": 1000}
                env = MujocoMulti(env_args=env_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario', type=str, default='Hopper-v2', help="Which mujoco task to run on")
    parser.add_argument('--agent_conf', type=str, default='3x1')
    parser.add_argument('--agent_obsk', type=int, default=0)
    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_true', default=False)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)

    # agent-specific state should be designed carefully
    parser.add_argument("--use_state_agent", action='store_true', default=False)
    parser.add_argument("--use_mustalive", action='store_false', default=True)
    parser.add_argument("--add_center_xy", action='store_true', default=False)
    parser.add_argument("--use_single_network", action='store_true', default=False)
    
    # 异步训练相关参数
    parser.add_argument("--async_mode", type=str, default='hierarchical', 
                       choices=['hierarchical', 'importance', 'progressive'],
                       help="异步训练模式")
    parser.add_argument("--async_ratio", type=float, default=0.7,
                       help="异步更新比例")
    parser.add_argument("--num_groups", type=int, default=4,
                       help="分层模式的组数")
    parser.add_argument("--importance_window", type=int, default=10,
                       help="重要性计算窗口大小")
    parser.add_argument("--use_async", action='store_true', default=False,
                       help="是否启用异步训练")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def setup_async_config(all_args):
    """
    根据命令行参数设置异步配置
    """
    if all_args.async_mode == 'hierarchical':
        async_config = HIERARCHICAL_CONFIG
    elif all_args.async_mode == 'importance':
        async_config = IMPORTANCE_CONFIG
    elif all_args.async_mode == 'progressive':
        async_config = PROGRESSIVE_CONFIG
    else:
        async_config = AsyncConfig()
    
    # 从命令行参数更新配置
    async_config.update_from_args(all_args)
    async_config.validate_config()
    
    # 将异步配置合并到all_args中
    for key, value in async_config.get_config_dict().items():
        setattr(all_args, key, value)
    
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    
    # 设置异步配置
    if all_args.use_async or all_args.algorithm_name == "async_mappo_lagr":
        all_args.use_async = True  # 确保异步标志被设置
        all_args = setup_async_config(all_args)
        print(f"使用异步训练模式: {all_args.async_mode}")
        print(f"异步比例: {all_args.async_ratio}")
    
    print("异步MAPPO-Lagrangian配置: ", all_args)

    if all_args.algorithm_name in ["mappo_lagr", "async_mappo_lagr"]:
        all_args.share_policy = False
    else:
        raise NotImplementedError

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("cuda flag: ", all_args.cuda, "Torch: ", torch.cuda.is_available())
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # 设置实验名称以区分异步和同步训练
    if all_args.use_async:
        experiment_suffix = f"_async_{all_args.async_mode}_{all_args.async_ratio}"
        all_args.experiment_name = all_args.experiment_name + experiment_suffix

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.scenario / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.experiment_name) +
                              "_seed" + str(all_args.seed),
                         group=all_args.map_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = envs.n_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from mappo_lagrangian.runner.shared.mujoco_runner import MujocoRunner as Runner
    else:
        if all_args.algorithm_name in ["mappo_lagr", "async_mappo_lagr"]:
            if all_args.use_async or all_args.algorithm_name == "async_mappo_lagr":
                # 使用异步训练器
                from mappo_lagrangian.runner.separated.async_mujoco_runner_mappo_lagr import AsyncMujocoRunner as Runner
                print("使用异步MAPPO-Lagrangian训练器")
            else:
                # 使用原始训练器
                from mappo_lagrangian.runner.separated.mujoco_runner_mappo_lagr import MujocoRunner as Runner
                print("使用原始MAPPO-Lagrangian训练器")
        else:
            from mappo_lagrangian.runner.separated.mujoco_runner import MujocoRunner as Runner

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])