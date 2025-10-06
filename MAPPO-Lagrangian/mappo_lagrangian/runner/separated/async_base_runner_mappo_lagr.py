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
            
        # 第三阶段：强制同步所有智能体的拉格朗日乘子
        self.sync_all_lagrangian_multipliers()
        
        # 方法执行完成，无需返回值
    
    def sync_all_lagrangian_multipliers(self):
        """
        强制同步所有智能体的拉格朗日乘子，确保全局一致性
        """
        if hasattr(self, 'trainer') and len(self.trainer) > 0:
            # 计算所有智能体拉格朗日乘子的平均值
            avg_lamda = np.mean([getattr(trainer, 'lamda_lagr', 0.1) for trainer in self.trainer])
            
            # 将平均值应用到所有智能体
            for trainer in self.trainer:
                trainer.lamda_lagr = avg_lamda
            
            # 记录同步信息
            if hasattr(self, 'logger'):
                self.logger.info(f"同步拉格朗日乘子: {avg_lamda:.6f}")
        
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
            
        return train_infos
    
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
            
        return train_infos
    
    def progressive_async_train(self):
        """
        改进的渐进式异步训练：契合MAPPO-Lagrangian Algorithm 3
        关键改进：
        1. 拉格朗日乘子全局一致性更新
        2. 基于安全约束违反程度的智能体重要性评估
        3. 渐进式同步策略保证训练稳定性
        """
        train_infos = []
        cost_train_infos = []
        
        # 计算训练进度和动态同步比例
        progress = getattr(self, 'current_episode', 0) / getattr(self, 'total_episodes', 1000)
        
        # 改进的同步比例计算：考虑安全约束满足情况
        base_sync_ratio = max(0.8, 1.0 - 0.3 * progress)  # 提高基础同步比例，增强稳定性
        safety_violation_penalty = self.calculate_safety_violation_penalty()
        adaptive_sync_ratio = min(1.0, base_sync_ratio + safety_violation_penalty)
        
        num_sync_agents = max(2, int(self.num_agents * adaptive_sync_ratio))
        
        # 基于安全重要性选择关键智能体
        critical_agents = self.select_safety_critical_agents(num_sync_agents)
        
        # 记录当前轮次的重要性
        self.agent_importance_history.append(np.zeros(self.num_agents))
        
        # 第一阶段：关键智能体同步更新（保证拉格朗日乘子一致性）
        critical_train_infos = self.update_critical_agents_sync(critical_agents)
        train_infos.extend(critical_train_infos)
        
        # 第二阶段：其他智能体异步更新（使用更新后的拉格朗日乘子）
        other_agents = [i for i in range(self.num_agents) if i not in critical_agents]
        if other_agents:
            other_train_infos = self.update_other_agents_async(other_agents)
            train_infos.extend(other_train_infos)
            
        return train_infos
    
    def calculate_safety_violation_penalty(self):
        """
        计算安全约束违反惩罚，用于动态调整同步比例
        """
        if not hasattr(self, 'recent_episode_costs') or len(self.recent_episode_costs) == 0:
            return 0.0
        
        # 计算最近几个episode的平均成本
        recent_costs = np.array(self.recent_episode_costs[-10:])  # 最近10个episode
        avg_cost = np.mean(recent_costs)
        
        # 如果超过安全边界，增加同步比例
        safety_bound = getattr(self, 'safety_bound', 0.2)
        if avg_cost > safety_bound:
            violation_ratio = (avg_cost - safety_bound) / safety_bound
            return min(0.3, violation_ratio * 0.5)  # 最多增加30%的同步比例
        
        return 0.0
    
    def select_safety_critical_agents(self, num_critical):
        """
        基于安全重要性选择关键智能体
        结合梯度范数、成本违反程度和策略变化幅度
        """
        importance_scores = np.zeros(self.num_agents)
        
        # 1. 基于梯度范数的重要性（原有机制）
        gradient_importance = self.calculate_agent_importance()
        
        # 2. 基于安全约束违反的重要性
        safety_importance = self.calculate_safety_importance()
        
        # 3. 基于策略变化幅度的重要性
        policy_change_importance = self.calculate_policy_change_importance()
        
        # 综合重要性评分（权重可调）
        for i in range(self.num_agents):
            importance_scores[i] = (
                0.4 * gradient_importance[i] +
                0.4 * safety_importance[i] +
                0.2 * policy_change_importance[i]
            )
        
        # 选择重要性最高的智能体
        critical_indices = np.argsort(importance_scores)[-num_critical:]
        return critical_indices.tolist()
    
    def calculate_safety_importance(self):
        """
        计算基于安全约束的智能体重要性
        """
        importance = np.ones(self.num_agents)
        
        if hasattr(self, 'agent_cost_history') and len(self.agent_cost_history) > 0:
            # 基于每个智能体的成本贡献计算重要性
            recent_costs = np.array(self.agent_cost_history[-5:])  # 最近5步
            if recent_costs.size > 0:
                agent_avg_costs = np.mean(recent_costs, axis=0)
                # 成本越高，重要性越高
                importance = agent_avg_costs / (np.sum(agent_avg_costs) + 1e-8)
        
        return importance
    
    def calculate_policy_change_importance(self):
        """
        计算基于策略变化幅度的重要性
        """
        importance = np.ones(self.num_agents)
        
        if hasattr(self, 'agent_policy_change_history') and len(self.agent_policy_change_history) > 0:
            # 基于策略参数变化幅度
            recent_changes = np.array(self.agent_policy_change_history[-3:])  # 最近3步
            if recent_changes.size > 0:
                agent_avg_changes = np.mean(recent_changes, axis=0)
                # 策略变化越大，重要性越高
                importance = agent_avg_changes / (np.sum(agent_avg_changes) + 1e-8)
        
    def update_critical_agents_sync(self, critical_agents):
        """
        关键智能体同步更新，确保拉格朗日乘子的全局一致性
        这是MAPPO-Lagrangian Algorithm 3的核心改进
        """
        train_infos = []
        
        # 第一步：收集所有关键智能体的成本信息用于拉格朗日乘子更新
        all_episode_costs = []
        all_cost_advantages = []
        all_imp_weights = []
        
        # 预计算所有关键智能体的重要性权重和成本优势
        for agent_id in critical_agents:
            self.trainer[agent_id].prep_training()
            
            # 获取该智能体的成本信息
            episode_costs = self.buffer[agent_id].episode_costs if hasattr(self.buffer[agent_id], 'episode_costs') else np.zeros(1)
            cost_advantages = self.buffer[agent_id].cost_returns[:-1] - self.buffer[agent_id].cost_preds[:-1]
            
            # 计算重要性权重（用于拉格朗日乘子更新）
            available_actions = None if self.buffer[agent_id].available_actions is None \
                else self.buffer[agent_id].available_actions[:-1].reshape(-1, *self.buffer[
                                                                                   agent_id].available_actions.shape[2:])
            
            _, action_log_probs = self.trainer[agent_id].policy.actor.evaluate_actions(
                self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            
            old_action_log_probs = self.buffer[agent_id].action_log_probs[:-1].reshape(-1, *self.buffer[agent_id].action_log_probs.shape[2:])
            imp_weights = torch.exp(action_log_probs - old_action_log_probs)
            
            all_episode_costs.append(episode_costs)
            all_cost_advantages.append(cost_advantages)
            all_imp_weights.append(imp_weights.detach().cpu().numpy())
        
        # 第二步：全局拉格朗日乘子更新（基于所有关键智能体的信息）
        if all_episode_costs:
            global_episode_costs = np.concatenate(all_episode_costs)
            global_cost_advantages = np.concatenate(all_cost_advantages, axis=1) if all_cost_advantages else np.array([])
            global_imp_weights = np.concatenate(all_imp_weights) if all_imp_weights else np.array([])
            
            # 计算全局拉格朗日乘子更新
            safety_bound = getattr(self, 'safety_bound', 0.2)
            gamma = getattr(self, 'gamma', 0.99)
            lagrangian_coef = getattr(self, 'lagrangian_coef_rate', 1e-4)
            
            if len(global_episode_costs) > 0 and len(global_cost_advantages) > 0:
                delta_lamda_lagr = -((global_episode_costs.mean() - safety_bound) * (1 - gamma) + 
                                   (global_imp_weights * global_cost_advantages.flatten()).mean())
                
                # 更新所有关键智能体的拉格朗日乘子
                for agent_id in critical_agents:
                    current_lamda = getattr(self.trainer[agent_id], 'lamda_lagr', 0.1)
                    new_lamda = max(0.0, current_lamda - delta_lamda_lagr * lagrangian_coef)
                    self.trainer[agent_id].lamda_lagr = new_lamda
        
        # 第三步：使用统一的拉格朗日乘子进行策略更新
        for agent_id in critical_agents:
            train_info = self.trainer[agent_id].train(self.buffer[agent_id], update_actor=True)
            
            # 记录重要性信息
            grad_norm = train_info.get('actor_grad_norm', 0)
            self.record_agent_importance(agent_id, grad_norm)
            
            # 记录策略变化（用于重要性评估）
            self.record_policy_change(agent_id, train_info)
            
            train_infos.append(train_info)
            self.buffer[agent_id].after_update()
        
        return train_infos
    
    def update_other_agents_async(self, other_agents):
        """
        其他智能体异步更新，使用关键智能体更新后的拉格朗日乘子
        """
        train_infos = []
        
        # 获取关键智能体的平均拉格朗日乘子作为参考
        if hasattr(self, 'trainer') and len(self.trainer) > 0:
            avg_lamda_lagr = np.mean([getattr(trainer, 'lamda_lagr', 0.1) for trainer in self.trainer])
        else:
            avg_lamda_lagr = 0.1
        
        for agent_id in other_agents:
            # 使用参考拉格朗日乘子
            self.trainer[agent_id].lamda_lagr = avg_lamda_lagr
            
            # 强制策略更新，确保训练一致性
            update_actor = True  # 移除随机性，始终更新策略
            train_info = self.trainer[agent_id].train(self.buffer[agent_id], update_actor=update_actor)
            
            # 记录重要性信息
            grad_norm = train_info.get('actor_grad_norm', 0)
            self.record_agent_importance(agent_id, grad_norm)
            
            train_infos.append(train_info)
            self.buffer[agent_id].after_update()
        
        return train_infos
    
    def record_policy_change(self, agent_id, train_info):
        """
        记录策略变化信息，用于重要性评估
        """
        if not hasattr(self, 'agent_policy_change_history'):
            self.agent_policy_change_history = []
        
        if len(self.agent_policy_change_history) == 0:
            self.agent_policy_change_history.append(np.zeros(self.num_agents))
        
        # 使用策略损失和梯度范数作为策略变化的指标
        policy_loss = train_info.get('policy_loss', 0)
        actor_grad_norm = train_info.get('actor_grad_norm', 0)
        policy_change = abs(policy_loss) + actor_grad_norm
        
        self.agent_policy_change_history[-1][agent_id] = policy_change
        
        # 保持历史记录在合理范围内
        if len(self.agent_policy_change_history) > 10:
            self.agent_policy_change_history = self.agent_policy_change_history[-10:]

    def lightweight_value_update(self, agent_id):
        """
        轻量级价值函数更新，仅更新价值网络而不更新策略网络
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
    
    def log_async_training_stats(self, critical_agents, other_agents, sync_ratio, safety_violation_penalty):
        """
        记录异步训练的统计信息和监控数据
        """
        if not hasattr(self, 'async_training_logs'):
            self.async_training_logs = []
        
        # 计算拉格朗日乘子的统计信息
        lamda_values = [getattr(self.trainer[i], 'lamda_lagr', 0.1) for i in range(self.num_agents)]
        lamda_mean = np.mean(lamda_values)
        lamda_std = np.std(lamda_values)
        
        # 计算智能体重要性分布
        if hasattr(self, 'agent_importance_history') and len(self.agent_importance_history) > 0:
            importance_variance = np.var(self.agent_importance_history[-1])
        else:
            importance_variance = 0.0
        
        log_entry = {
            'episode': getattr(self, 'episode', 0),
            'sync_ratio': sync_ratio,
            'safety_violation_penalty': safety_violation_penalty,
            'num_critical_agents': len(critical_agents),
            'num_other_agents': len(other_agents),
            'lamda_mean': lamda_mean,
            'lamda_std': lamda_std,
            'importance_variance': importance_variance,
            'critical_agents': critical_agents.copy() if isinstance(critical_agents, list) else list(critical_agents)
        }
        
        self.async_training_logs.append(log_entry)
        
        # 保持日志在合理大小
        if len(self.async_training_logs) > 1000:
            self.async_training_logs = self.async_training_logs[-1000:]
        
        # 可选：打印关键信息
        if getattr(self, 'episode', 0) % 100 == 0:  # 每100个episode打印一次
            print(f"Episode {getattr(self, 'episode', 0)}: "
                  f"Sync Ratio: {sync_ratio:.3f}, "
                  f"Critical Agents: {len(critical_agents)}, "
                  f"Lambda Mean: {lamda_mean:.4f}±{lamda_std:.4f}")
    
    def get_async_training_summary(self):
        """
        获取异步训练的总结统计信息
        """
        if not hasattr(self, 'async_training_logs') or len(self.async_training_logs) == 0:
            return {}
        
        logs = self.async_training_logs
        recent_logs = logs[-100:] if len(logs) >= 100 else logs  # 最近100个episode
        
        summary = {
            'total_episodes': len(logs),
            'avg_sync_ratio': np.mean([log['sync_ratio'] for log in recent_logs]),
            'avg_safety_penalty': np.mean([log['safety_violation_penalty'] for log in recent_logs]),
            'avg_critical_agents': np.mean([log['num_critical_agents'] for log in recent_logs]),
            'lamda_convergence': {
                'mean': np.mean([log['lamda_mean'] for log in recent_logs]),
                'std': np.mean([log['lamda_std'] for log in recent_logs])
            },
            'importance_stability': np.mean([log['importance_variance'] for log in recent_logs])
        }
        
        return summary

    def train(self):
        """
        主训练函数，根据配置选择异步策略
        """
        if self.async_mode == 'hierarchical':
            return self.hierarchical_async_train()
        elif self.async_mode == 'importance':
            return self.importance_sampling_async_train()
        elif self.async_mode == 'progressive':
            return self.progressive_async_train()
        else:
            # 回退到原始同步训练
            return super().train()