import copy
import time
import wandb
import os
import numpy as np
from itertools import chain
import torch
from tensorboardX import SummaryWriter

from mappo_lagrangian.utils.separated_buffer import SeparatedReplayBuffer
from mappo_lagrangian.utils.util import update_linear_schedule
from mappo_lagrangian.runner.separated.base_runner_mappo_lagr import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class AsyncRunner(Runner):
    """
    改进的异步MAPPO-Lagrangian训练器
    实现多种异步策略来提高训练效率同时保持性能
    """
    
    def __init__(self, config):
        super(AsyncRunner, self).__init__(config)
        
        # 初始化trainer和buffer
        self.trainer = []
        self.buffer = []
        # 导入异步算法
        from mappo_lagrangian.algorithms.r_mappo_lagr.async_r_mappo_lagr import Async_R_MAPPO_Lagr as TrainAlgo
        
        for agent_id in range(self.num_agents):
            # algorithm - 使用异步版本
            tr = TrainAlgo(self.all_args, self.policy[agent_id], device=self.device)
            # buffer
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else \
            self.envs.observation_space[agent_id]
            bu = SeparatedReplayBuffer(self.all_args,
                                       self.envs.observation_space[agent_id],
                                       share_observation_space,
                                       self.envs.action_space[agent_id])
            self.buffer.append(bu)
            self.trainer.append(tr)
        
        # 异步训练相关参数
        self.async_mode = getattr(config, 'async_mode', 'hierarchical')  # 'hierarchical', 'importance', 'progressive'
        self.async_ratio = getattr(config, 'async_ratio', 0.7)  # 异步更新比例
        self.num_groups = getattr(config, 'num_groups', 4)  # 分层模式的组数
        self.importance_window = getattr(config, 'importance_window', 10)  # 重要性计算窗口
        
        # 用于跟踪智能体重要性的历史信息
        self.agent_importance_history = []
        self.agent_performance_history = []
        
        # 全局状态缓存，用于异步更新时的信息共享
        self.global_state_cache = {}
        
    def create_agent_groups(self, num_groups):
        """
        创建智能体分组，尽量平衡每组的能力
        """
        agents_per_group = self.num_agents // num_groups
        groups = []
        
        for i in range(num_groups):
            start_idx = i * agents_per_group
            if i == num_groups - 1:  # 最后一组包含剩余的所有智能体
                end_idx = self.num_agents
            else:
                end_idx = (i + 1) * agents_per_group
            groups.append(list(range(start_idx, end_idx)))
            
        return groups
    
    def calculate_agent_importance(self):
        """
        计算每个智能体的重要性分数
        基于：1) 策略梯度幅度 2) 约束违反程度 3) 性能贡献
        """
        importance_scores = np.ones(self.num_agents)
        
        if len(self.agent_importance_history) > 0:
            # 基于历史梯度幅度
            recent_grads = self.agent_importance_history[-self.importance_window:]
            grad_importance = np.mean(recent_grads, axis=0)
            
            # 基于性能贡献
            if len(self.agent_performance_history) > 0:
                recent_perf = self.agent_performance_history[-self.importance_window:]
                perf_importance = np.std(recent_perf, axis=0)  # 方差大的智能体更重要
                
                # 综合重要性分数
                importance_scores = 0.6 * grad_importance + 0.4 * perf_importance
                
        # 归一化
        importance_scores = importance_scores / (np.sum(importance_scores) + 1e-8)
        return importance_scores
    
    def importance_based_sampling(self, importance_scores, sample_ratio):
        """
        基于重要性分数采样智能体
        """
        num_selected = max(1, int(self.num_agents * sample_ratio))
        
        # 使用重要性分数作为采样概率
        selected_agents = np.random.choice(
            self.num_agents, 
            size=num_selected, 
            replace=False, 
            p=importance_scores
        )
        
        return selected_agents.tolist()
    
    def select_critical_agents(self, num_critical):
        """
        选择关键智能体（用于渐进式异步更新）
        """
        importance_scores = self.calculate_agent_importance()
        critical_indices = np.argsort(importance_scores)[-num_critical:]
        return critical_indices.tolist()
    
    def update_agent_group(self, agent_ids, group_id):
        """
        更新指定组的智能体
        """
        train_infos = []
        
        # 计算组内因子
        action_dim = self.buffer[0].actions.shape[-1]
        group_factor = np.ones((self.episode_length, self.n_rollout_threads, action_dim), dtype=np.float32)
        
        for agent_id in agent_ids:
            self.trainer[agent_id].prep_training()
            self.buffer[agent_id].update_factor(group_factor)
            
            # 计算旧策略的log概率
            available_actions = None if self.buffer[agent_id].available_actions is None \
                else self.buffer[agent_id].available_actions[:-1].reshape(-1, *self.buffer[
                                                                                   agent_id].available_actions.shape[2:])
            
            old_actions_logprob, _ = self.trainer[agent_id].policy.actor.evaluate_actions(
                self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            
            # 训练智能体
            train_info = self.trainer[agent_id].train(self.buffer[agent_id])
            
            # 计算新策略的log概率
            new_actions_logprob, _ = self.trainer[agent_id].policy.actor.evaluate_actions(
                self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            
            # 更新组内因子
            group_factor = group_factor * _t2n(torch.exp(new_actions_logprob - old_actions_logprob).reshape(
                self.episode_length, self.n_rollout_threads, action_dim))
            
            # 记录重要性信息
            grad_norm = train_info.get('actor_grad_norm', 0)
            self.record_agent_importance(agent_id, grad_norm)
            
            train_infos.append(train_info)
            self.buffer[agent_id].after_update()
            
        return train_infos
    
    def record_agent_importance(self, agent_id, grad_norm):
        """
        记录智能体的重要性信息
        """
        if len(self.agent_importance_history) == 0:
            self.agent_importance_history.append(np.zeros(self.num_agents))
            
        self.agent_importance_history[-1][agent_id] = grad_norm
        
        # 保持历史记录在合理范围内
        if len(self.agent_importance_history) > self.importance_window * 2:
            self.agent_importance_history = self.agent_importance_history[-self.importance_window:]
    
    def hierarchical_async_train(self):
        """
        分层异步训练：将智能体分组，组内同步，组间异步
        """
        train_infos = []
        cost_train_infos = []
        
        # 创建智能体分组
        agent_groups = self.create_agent_groups(self.num_groups)
        
        # 记录当前轮次的重要性
        self.agent_importance_history.append(np.zeros(self.num_agents))
        
        # 组间异步更新
        for group_id, agent_ids in enumerate(agent_groups):
            group_train_infos = self.update_agent_group(agent_ids, group_id)
            train_infos.extend(group_train_infos)
            
        return train_infos, cost_train_infos
    
    def importance_sampling_async_train(self):
        """
        基于重要性采样的异步训练
        """
        train_infos = []
        cost_train_infos = []
        
        # 计算智能体重要性并采样
        importance_scores = self.calculate_agent_importance()
        selected_agents = self.importance_based_sampling(importance_scores, self.async_ratio)
        
        # 记录当前轮次的重要性
        self.agent_importance_history.append(np.zeros(self.num_agents))
        
        # 更新选中的智能体
        selected_train_infos = self.update_agent_group(selected_agents, 0)
        train_infos.extend(selected_train_infos)
        
        # 对未选中的智能体进行轻量级更新（仅更新价值函数）
        unselected_agents = [i for i in range(self.num_agents) if i not in selected_agents]
        for agent_id in unselected_agents:
            # 仅更新价值函数，保持策略一致性
            self.lightweight_value_update(agent_id)
            
        return train_infos, cost_train_infos
    
    def progressive_async_train(self):
        """
        渐进式异步训练：根据训练进度调整同步程度
        """
        train_infos = []
        cost_train_infos = []
        
        # 计算当前的同步比例（训练初期高，后期低）
        progress = getattr(self, 'current_episode', 0) / getattr(self, 'total_episodes', 1000)
        sync_ratio = max(0.3, 1.0 - progress)
        num_sync_agents = max(1, int(self.num_agents * sync_ratio))
        
        # 选择关键智能体进行同步更新
        critical_agents = self.select_critical_agents(num_sync_agents)
        
        # 记录当前轮次的重要性
        self.agent_importance_history.append(np.zeros(self.num_agents))
        
        # 关键智能体同步更新
        critical_train_infos = self.update_agent_group(critical_agents, 0)
        train_infos.extend(critical_train_infos)
        
        # 其他智能体异步更新
        other_agents = [i for i in range(self.num_agents) if i not in critical_agents]
        if other_agents:
            other_train_infos = self.update_agent_group(other_agents, 1)
            train_infos.extend(other_train_infos)
            
        return train_infos, cost_train_infos
    
    def lightweight_value_update(self, agent_id):
        """
        轻量级价值函数更新，用于未选中的智能体
        """
        self.trainer[agent_id].prep_training()
        
        # 仅更新价值函数和成本价值函数
        for _ in range(2):  # 减少更新次数
            if self._use_naive_recurrent:
                data_generator = self.buffer[agent_id].naive_recurrent_generator(
                    self.buffer[agent_id].returns[:-1] - self.buffer[agent_id].value_preds[:-1], 
                    1,  # 只用一个mini-batch
                    self.buffer[agent_id].cost_returns[:-1] - self.buffer[agent_id].cost_preds[:-1]
                )
            else:
                data_generator = self.buffer[agent_id].feed_forward_generator(
                    self.buffer[agent_id].returns[:-1] - self.buffer[agent_id].value_preds[:-1], 
                    1,
                    cost_adv=self.buffer[agent_id].cost_returns[:-1] - self.buffer[agent_id].cost_preds[:-1]
                )
                
            for sample in data_generator:
                # 仅更新价值函数，不更新策略
                self.trainer[agent_id].ppo_update(sample, update_actor=False)
                break  # 只用第一个batch
    
    def train(self):
        """
        ========== 算法3 MAPPO-Lagrangian 异步训练策略选择 ==========
        主训练函数，根据配置选择不同的异步策略来优化训练效率
        
        异步策略说明：
        - hierarchical: 分层异步更新，将智能体分组进行异步训练
        - importance: 基于重要性采样的异步更新，优先训练重要智能体
        - progressive: 渐进式异步更新，随训练进程动态调整异步比例
        """
        if self.async_mode == 'hierarchical':
            # ========== 算法3 异步策略1: 分层异步训练 ==========
            # 将智能体分为不同层级，组内同步更新，组间异步更新
            return self.hierarchical_async_train()
        elif self.async_mode == 'importance':
            # ========== 算法3 异步策略2: 重要性采样异步训练 ==========
            # 基于智能体重要性分数进行采样，优先更新关键智能体
            return self.importance_sampling_async_train()
        elif self.async_mode == 'progressive':
            # ========== 算法3 异步策略3: 渐进式异步训练 ==========
            # 随训练进程动态调整异步比例，平衡效率与性能
            return self.progressive_async_train()
        else:
            # ========== 算法3 回退策略: 同步训练 ==========
            # 当异步模式未指定或不支持时，回退到原始同步训练
            return super().train()