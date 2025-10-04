"""
MAPPO-Lagrangian算法的Mujoco环境运行器
该文件实现了在Mujoco物理仿真环境中运行MAPPO-Lagrangian算法的核心逻辑
包括训练循环、数据收集、环境交互、模型评估等功能
"""

# 时间相关模块，用于计算训练时间和FPS
import time
from itertools import chain

import wandb
import numpy as np
from functools import reduce
import torch
# 导入基础运行器类，MujocoRunner继承自该类
from mappo_lagrangian.runner.separated.base_runner_mappo_lagr import Runner


def _t2n(x):
    """
    工具函数：将PyTorch张量转换为NumPy数组
    
    Args:
        x (torch.Tensor): 输入的PyTorch张量
        
    Returns:
        numpy.ndarray: 转换后的NumPy数组，已从GPU移到CPU并分离梯度
        
    功能说明：
    - detach(): 从计算图中分离张量，移除梯度信息
    - cpu(): 将张量从GPU移动到CPU
    - numpy(): 将PyTorch张量转换为NumPy数组
    """
    return x.detach().cpu().numpy()


class MujocoRunner(Runner):
    """
    Mujoco环境的MAPPO-Lagrangian算法运行器
    
    该类继承自基础Runner类，专门用于在Mujoco物理仿真环境中执行：
    - 多智能体强化学习训练
    - 约束优化（通过拉格朗日乘子）
    - 数据收集和环境交互
    - 模型评估和性能监控
    
    主要功能：
    1. 训练循环管理
    2. 环境交互和数据收集
    3. 经验缓冲区管理
    4. 模型训练和更新
    5. 性能评估和日志记录
    """

    def __init__(self, config):
        """
        初始化MujocoRunner
        
        Args:
            config: 配置对象，包含所有训练参数和环境设置
            
        功能：
        - 调用父类初始化方法
        - 继承所有基础配置和组件（环境、策略、缓冲区等）
        """
        super(MujocoRunner, self).__init__(config)

    def run(self):
        """
        主训练循环方法 - MAPPO-Lagrangian算法的核心执行流程
        
        该方法实现了完整的训练流程，包括：
        1. 环境预热 (warmup)
        2. 多轮训练循环
        3. 数据收集与环境交互
        4. 策略更新与约束优化
        5. 性能监控与模型保存
        6. 定期评估
        
        训练流程对应论文中的算法框架：
        - 外层循环：多个训练episode
        - 内层循环：每个episode内的环境步骤
        - 每个episode结束后进行策略更新
        """
        # 环境预热：重置环境状态，初始化观测和缓冲区
        self.warmup()

        # 记录训练开始时间，用于计算FPS
        start = time.time()
        
        # 计算总训练轮数 = 总环境步数 / 每轮步数 / 并行环境数
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        # 初始化每个并行环境的累积奖励和成本跟踪
        # train_episode_rewards: 跟踪每个环境当前episode的累积奖励
        train_episode_rewards = [0 for _ in range(self.n_rollout_threads)]
        # train_episode_costs: 跟踪每个环境当前episode的累积约束成本
        train_episode_costs = [0 for _ in range(self.n_rollout_threads)]

        # 主训练循环：遍历所有训练episode
        for episode in range(episodes):
            # 线性学习率衰减：随着训练进行逐渐降低学习率
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            # 存储本轮完成的episode的奖励和成本
            done_episodes_rewards = []
            done_episodes_costs = []

            # 内层循环：在当前episode中执行环境步骤
            for step in range(self.episode_length):
                # === 数据收集阶段 ===
                # 调用collect方法收集当前步骤的所有必要数据
                # 返回值包括：价值估计、动作、动作对数概率、RNN状态、成本预测等
                values, actions, action_log_probs, rnn_states, rnn_states_critic, cost_preds, \
                rnn_states_cost = self.collect(step)

                # === 环境交互阶段 ===
                # 执行动作，获取环境反馈
                # obs: 新的观测状态
                # share_obs: 共享观测（用于中心化价值函数）
                # rewards: 奖励信号
                # costs: 约束成本信号（MAPPO-Lagrangian的关键）
                # dones: 环境终止标志
                # infos: 额外信息
                obs, share_obs, rewards, costs, dones, infos, _ = self.envs.step(actions)

                # === 奖励和成本统计 ===
                # 检查哪些环境已完成episode（所有智能体都done）
                dones_env = np.all(dones, axis=1)
                # 计算每个环境的平均奖励（跨所有智能体）
                reward_env = np.mean(rewards, axis=1).flatten()
                # 计算每个环境的平均成本（跨所有智能体）
                cost_env = np.mean(costs, axis=1).flatten()
                
                # 累积当前步骤的奖励和成本到episode总计中
                train_episode_rewards += reward_env
                train_episode_costs += cost_env
                
                # 处理完成的episode：记录最终奖励和成本，重置计数器
                for t in range(self.n_rollout_threads):
                    if dones_env[t]:
                        # 记录完成episode的总奖励和成本
                        done_episodes_rewards.append(train_episode_rewards[t])
                        train_episode_rewards[t] = 0
                        done_episodes_costs.append(train_episode_costs[t])
                        train_episode_costs[t] = 0

                # === 数据打包 ===
                # 将所有收集的数据打包，准备插入经验缓冲区
                data = obs, share_obs, rewards, costs, dones, infos, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic, cost_preds, rnn_states_cost

                # === 数据存储 ===
                # 将经验数据插入到每个智能体的经验缓冲区中
                self.insert(data)   

            # === 训练更新阶段 ===
            # episode结束后，计算回报和优势函数
            self.compute()
            # 执行策略更新和约束优化（对应论文算法3）
            train_infos = self.train()

            # === 后处理阶段 ===
            # 计算当前总训练步数
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # === 模型保存 ===
            # 定期保存模型检查点
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # === 日志记录 ===
            # 定期记录训练信息和性能指标
            if episode % self.log_interval == 0:
                end = time.time()
                # 打印训练进度信息
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.scenario,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                # 记录训练信息到日志
                self.log_train(train_infos, total_num_steps)

                # 如果有完成的episode，记录平均奖励和成本
                if len(done_episodes_rewards) > 0:
                    aver_episode_rewards = np.mean(done_episodes_rewards)
                    aver_episode_costs = np.mean(done_episodes_costs)
                    # 更新平均成本到缓冲区（用于约束优化）
                    self.return_aver_cost(aver_episode_costs)
                    print("some episodes done, average rewards: {}, average costs: {}".format(aver_episode_rewards,
                                                                                              aver_episode_costs))
                    # 记录到TensorBoard/WandB
                    self.writter.add_scalars("train_episode_rewards", {"aver_rewards": aver_episode_rewards},
                                             total_num_steps)
                    self.writter.add_scalars("train_episode_costs", {"aver_costs": aver_episode_costs},
                                             total_num_steps)

            # === 模型评估 ===
            # 定期进行模型性能评估
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def return_aver_cost(self, aver_episode_costs):
        """
        将平均episode成本更新到所有智能体的经验缓冲区
        
        Args:
            aver_episode_costs (float): 平均episode成本
            
        功能：
        - 为每个智能体的缓冲区插入平均成本信息
        - 用于MAPPO-Lagrangian算法的约束优化
        - 帮助拉格朗日乘子的更新计算
        """
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].return_aver_insert(aver_episode_costs)

    def warmup(self):
        """
        环境预热方法 - 初始化训练前的环境状态
        
        功能：
        1. 重置所有并行环境到初始状态
        2. 获取初始观测状态
        3. 将初始观测存储到每个智能体的经验缓冲区
        4. 为训练循环做好准备
        
        注意：
        - 如果不使用中心化价值函数，则共享观测等于个体观测
        - 这是训练开始前的必要步骤
        """
        # 重置所有并行环境，获取初始观测
        # obs: 每个智能体的个体观测 [n_envs, n_agents, obs_dim]
        # share_obs: 用于中心化价值函数的共享观测 [n_envs, n_agents, share_obs_dim]
        obs, share_obs, _ = self.envs.reset()
        
        # 如果不使用中心化价值函数，共享观测就是个体观测
        if not self.use_centralized_V:
            share_obs = obs

        # 为每个智能体初始化缓冲区的第0步观测
        for agent_id in range(self.num_agents):
            # 存储共享观测到缓冲区（用于价值函数）
            self.buffer[agent_id].share_obs[0] = share_obs[:, agent_id].copy()
            # 存储个体观测到缓冲区（用于策略网络）
            self.buffer[agent_id].obs[0] = obs[:, agent_id].copy()

    @torch.no_grad()
    def collect(self, step):
        """
        数据收集方法 - 从所有智能体收集动作和状态信息
        
        Args:
            step (int): 当前环境步骤索引
            
        Returns:
            tuple: 包含以下元素的元组
                - values: 价值函数估计 [n_envs, n_agents, 1]
                - actions: 选择的动作 [n_envs, n_agents, action_dim]
                - action_log_probs: 动作的对数概率 [n_envs, n_agents, 1]
                - rnn_states: 策略网络的RNN状态 [n_envs, n_agents, hidden_size]
                - rnn_states_critic: 价值网络的RNN状态 [n_envs, n_agents, hidden_size]
                - cost_preds: 成本预测 [n_envs, n_agents, 1]
                - rnn_states_cost: 成本网络的RNN状态 [n_envs, n_agents, hidden_size]
        
        功能详解：
        1. 遍历所有智能体，为每个智能体生成动作
        2. 使用当前观测、RNN状态和mask信息
        3. 调用策略网络的get_actions方法
        4. 收集价值估计、动作、概率、成本预测等信息
        5. 将数据从PyTorch张量转换为NumPy数组
        6. 重新排列数据维度以匹配环境格式
        
        注意：
        - 使用@torch.no_grad()装饰器避免梯度计算，提高效率
        - 数据收集阶段不需要梯度，只在训练阶段需要
        """
        # 初始化收集器列表，用于存储每个智能体的输出
        value_collector = []           # 价值函数估计
        action_collector = []          # 动作
        action_log_prob_collector = [] # 动作对数概率
        rnn_state_collector = []       # 策略网络RNN状态
        rnn_state_critic_collector = [] # 价值网络RNN状态
        cost_preds_collector = []      # 成本预测
        rnn_states_cost_collector = [] # 成本网络RNN状态

        # 遍历所有智能体，为每个智能体生成动作和相关信息
        for agent_id in range(self.num_agents):
            # 设置智能体为rollout模式（推理模式，不计算梯度）
            self.trainer[agent_id].prep_rollout()
            
            # 调用智能体策略网络的get_actions方法
            # 输入：当前步骤的观测、RNN状态、mask等
            # 输出：价值估计、动作、动作概率、新RNN状态、成本预测等
            value, action, action_log_prob, rnn_state, rnn_state_critic, cost_pred, rnn_state_cost \
                = self.trainer[agent_id].policy.get_actions(
                    self.buffer[agent_id].share_obs[step],    # 共享观测（用于价值函数）
                    self.buffer[agent_id].obs[step],          # 个体观测（用于策略网络）
                    self.buffer[agent_id].rnn_states[step],  # 策略网络RNN状态
                    self.buffer[agent_id].rnn_states_critic[step], # 价值网络RNN状态
                    self.buffer[agent_id].masks[step],       # mask（处理episode结束）
                    rnn_states_cost=self.buffer[agent_id].rnn_states_cost[step] # 成本网络RNN状态
                )
            
            # 将PyTorch张量转换为NumPy数组并收集
            value_collector.append(_t2n(value))
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
            rnn_state_critic_collector.append(_t2n(rnn_state_critic))
            cost_preds_collector.append(_t2n(cost_pred))
            rnn_states_cost_collector.append(_t2n(rnn_state_cost))
            
        # 数据维度重排：从 [agents, envs, dim] 转换为 [envs, agents, dim]
        # 这样的格式更适合环境交互，因为环境期望 [n_envs, n_agents, ...] 的格式
        values = np.array(value_collector).transpose(1, 0, 2)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_state_critic_collector).transpose(1, 0, 2, 3)
        cost_preds = np.array(cost_preds_collector).transpose(1, 0, 2)
        rnn_states_cost = np.array(rnn_states_cost_collector).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, cost_preds, rnn_states_cost

    def insert(self, data):
        """
        数据插入方法 - 将收集的经验数据插入到经验缓冲区
        
        Args:
            data (tuple): 包含所有经验数据的元组，包括：
                - obs: 新观测状态 [n_envs, n_agents, obs_dim]
                - share_obs: 共享观测状态 [n_envs, n_agents, share_obs_dim]
                - rewards: 奖励信号 [n_envs, n_agents, 1]
                - costs: 约束成本信号 [n_envs, n_agents, 1]
                - dones: 环境终止标志 [n_envs, n_agents, 1]
                - infos: 额外信息
                - values: 价值函数估计 [n_envs, n_agents, 1]
                - actions: 执行的动作 [n_envs, n_agents, action_dim]
                - action_log_probs: 动作对数概率 [n_envs, n_agents, 1]
                - rnn_states: 策略网络RNN状态 [n_envs, n_agents, hidden_size]
                - rnn_states_critic: 价值网络RNN状态 [n_envs, n_agents, hidden_size]
                - cost_preds: 成本预测 [n_envs, n_agents, 1]
                - rnn_states_cost: 成本网络RNN状态 [n_envs, n_agents, hidden_size]
        
        功能详解：
        1. 解包经验数据
        2. 处理环境终止状态，重置相应的RNN状态
        3. 创建mask和active_mask来处理episode边界
        4. 将数据插入到每个智能体的经验缓冲区
        
        关键处理：
        - 当环境终止时，重置RNN状态为零
        - 使用mask标记有效的时间步
        - 使用active_mask标记活跃的智能体
        """
        # 解包经验数据元组
        obs, share_obs, rewards, costs, dones, infos, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic, cost_preds, rnn_states_cost = data
        
        # 检查哪些环境已完成episode（所有智能体都done）
        dones_env = np.all(dones, axis=1)

        # === RNN状态重置处理 ===
        # 当环境终止时，需要重置RNN状态，避免跨episode的状态污染
        
        # 重置策略网络的RNN状态
        rnn_states[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        
        # 重置价值网络的RNN状态
        rnn_states_critic[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_critic.shape[2:]), dtype=np.float32)

        # 重置成本网络的RNN状态
        rnn_states_cost[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_cost.shape[2:]), dtype=np.float32)

        # === Mask创建 ===
        # masks用于标记有效的时间步（1表示有效，0表示episode结束）
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # 环境终止的位置设置为0
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # active_masks用于标记活跃的智能体
        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # 个别智能体终止的位置设置为0
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        # 但是如果整个环境终止，所有智能体都重新激活
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # === 共享观测处理 ===
        # 如果不使用中心化价值函数，共享观测就是个体观测
        if not self.use_centralized_V:
            share_obs = obs

        # === 数据插入到缓冲区 ===
        # 为每个智能体将经验数据插入到对应的缓冲区中
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(
                share_obs[:, agent_id],           # 共享观测
                obs[:, agent_id],                 # 个体观测
                rnn_states[:, agent_id],          # 策略网络RNN状态
                rnn_states_critic[:, agent_id],   # 价值网络RNN状态
                actions[:, agent_id],             # 动作
                action_log_probs[:, agent_id],    # 动作对数概率
                values[:, agent_id],              # 价值估计
                rewards[:, agent_id],             # 奖励
                masks[:, agent_id],               # mask
                None,                             # bad_masks（未使用）
                active_masks[:, agent_id],        # active_masks
                None,                             # available_actions（未使用）
                costs=costs[:, agent_id],         # 约束成本
                cost_preds=cost_preds[:, agent_id], # 成本预测
                rnn_states_cost=rnn_states_cost[:, agent_id] # 成本网络RNN状态
            )

    def log_train(self, train_infos, total_num_steps):
        """
        训练日志记录方法 - 记录和可视化训练过程中的关键指标
        
        Args:
            train_infos (list): 训练信息列表，包含每个智能体的训练统计
            total_num_steps (int): 当前总训练步数
            
        功能：
        1. 计算和打印平均步骤奖励
        2. 为每个智能体记录详细的训练指标
        3. 将指标发送到WandB或TensorBoard进行可视化
        
        记录的指标包括：
        - 平均步骤奖励
        - 策略损失、价值损失
        - 拉格朗日乘子相关指标
        - 其他算法特定的统计信息
        """
        # 打印当前缓冲区中的平均步骤奖励
        print("average_step_rewards is {}.".format(np.mean(self.buffer[0].rewards)))
        
        # 初始化第一个智能体的平均步骤奖励
        train_infos[0][0]["average_step_rewards"] = 0
        
        # 为每个智能体记录训练指标
        for agent_id in range(self.num_agents):
            # 计算当前智能体的平均步骤奖励
            train_infos[0][agent_id]["average_step_rewards"] = np.mean(self.buffer[agent_id].rewards)
            
            # 遍历该智能体的所有训练指标
            for k, v in train_infos[0][agent_id].items():
                # 为指标添加智能体前缀
                agent_k = "agent%i/" % agent_id + k
                
                # 根据配置选择日志记录方式
                if self.use_wandb:
                    # 使用Weights & Biases记录
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    # 使用TensorBoard记录
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        """
        模型评估方法 - 在独立的评估环境中测试当前策略的性能
        
        Args:
            total_num_steps (int): 当前总训练步数，用于日志记录
            
        功能详解：
        1. 在评估环境中运行多个episode
        2. 使用确定性策略（不添加探索噪声）
        3. 收集episode奖励和成本统计
        4. 记录评估结果到日志系统
        
        评估特点：
        - 使用@torch.no_grad()避免梯度计算
        - 策略采用确定性模式（deterministic=True）
        - 不更新模型参数，仅用于性能测试
        - 独立于训练环境，避免影响训练数据
        
        评估流程：
        1. 初始化评估环境和统计变量
        2. 循环运行评估episode
        3. 收集每个episode的奖励和成本
        4. 计算平均性能指标
        5. 记录结果并打印统计信息
        """
        # === 初始化评估统计变量 ===
        eval_episode = 0  # 已完成的评估episode计数
        eval_episode_rewards = []  # 存储每个评估环境的episode奖励
        one_episode_rewards = []   # 存储当前episode的步骤奖励
        eval_episode_costs = []    # 存储每个评估环境的episode成本
        one_episode_costs = []     # 存储当前episode的步骤成本

        # 为每个评估环境初始化奖励和成本收集器
        for eval_i in range(self.n_eval_rollout_threads):
            one_episode_rewards.append([])
            eval_episode_rewards.append([])
            one_episode_costs.append([])
            eval_episode_costs.append([])

        # === 重置评估环境 ===
        # 获取初始观测状态
        eval_obs, eval_share_obs, _ = self.eval_envs.reset()

        # === 初始化评估状态 ===
        # 初始化RNN状态（用于循环神经网络策略）
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                   dtype=np.float32)
        # 初始化mask（全部设为1，表示所有状态都有效）
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        # === 主评估循环 ===
        while True:
            # === 动作选择阶段 ===
            eval_actions_collector = []  # 收集所有智能体的动作
            eval_rnn_states_collector = []  # 收集更新后的RNN状态
            
            # 为每个智能体生成动作
            for agent_id in range(self.num_agents):
                # 设置智能体为rollout模式（推理模式）
                self.trainer[agent_id].prep_rollout()
                
                # 使用确定性策略生成动作（不添加探索噪声）
                # 这确保评估结果的一致性和可重复性
                eval_actions, temp_rnn_state = \
                    self.trainer[agent_id].policy.act(
                        eval_obs[:, agent_id],        # 当前观测
                        eval_rnn_states[:, agent_id], # 当前RNN状态
                        eval_masks[:, agent_id],      # mask
                        deterministic=True            # 确定性策略
                    )
                
                # 更新RNN状态并转换为NumPy数组
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))

            # 重新排列动作维度：从[agents, envs, action_dim]到[envs, agents, action_dim]
            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

            # === 环境交互阶段 ===
            # 在评估环境中执行动作，获取反馈
            eval_obs, eval_share_obs, eval_rewards, eval_costs, eval_dones, eval_infos, _ = self.eval_envs.step(
                eval_actions)
            
            # === 奖励和成本收集 ===
            # 将当前步骤的奖励和成本添加到episode统计中
            for eval_i in range(self.n_eval_rollout_threads):
                one_episode_rewards[eval_i].append(eval_rewards[eval_i])
                one_episode_costs[eval_i].append(eval_costs[eval_i])

            # === Episode终止处理 ===
            # 检查哪些评估环境已完成episode
            eval_dones_env = np.all(eval_dones, axis=1)

            # 重置已完成环境的RNN状态
            eval_rnn_states[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            # 更新mask：完成的环境设为0，其他保持1
            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1),
                                                          dtype=np.float32)

            # === Episode完成统计 ===
            # 处理已完成的episode
            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    # 计算episode总奖励（所有步骤奖励的累积）
                    eval_episode_rewards[eval_i].append(np.sum(one_episode_rewards[eval_i], axis=0))
                    # 重置当前episode的奖励收集器
                    one_episode_rewards[eval_i] = []

            # === 评估完成检查 ===
            # 如果已完成足够数量的评估episode，结束评估
            if eval_episode >= self.all_args.eval_episodes:
                # 合并所有环境的episode奖励
                eval_episode_rewards = np.concatenate(eval_episode_rewards)
                
                # === 评估结果统计 ===
                # 创建评估信息字典
                eval_env_infos = {
                    'eval_average_episode_rewards': eval_episode_rewards,
                    'eval_max_episode_rewards': [np.max(eval_episode_rewards)]
                }
                
                # === 日志记录 ===
                # 将评估结果记录到日志系统
                self.log_env(eval_env_infos, total_num_steps)
                
                # 打印评估结果
                print("eval_average_episode_rewards is {}.".format(np.mean(eval_episode_rewards)))
                
                # 结束评估循环
                break
