"""
异步训练的工具函数
"""
import numpy as np
import torch
from typing import List, Tuple, Dict, Any

def calculate_agent_importance(trainer_list: List, buffer_list: List, 
                             episode_rewards: np.ndarray, 
                             episode_costs: np.ndarray) -> np.ndarray:
    """
    计算智能体的重要性分数，用于重要性采样异步更新
    
    Args:
        trainer_list: 训练器列表
        buffer_list: 缓冲区列表
        episode_rewards: 每个智能体的奖励
        episode_costs: 每个智能体的成本
    
    Returns:
        importance_scores: 重要性分数数组
    """
    num_agents = len(trainer_list)
    importance_scores = np.zeros(num_agents)
    
    for agent_id in range(num_agents):
        # 基于奖励和成本的重要性
        reward_importance = np.abs(episode_rewards[agent_id]) if episode_rewards[agent_id] != 0 else 0.1
        cost_importance = np.abs(episode_costs[agent_id]) if episode_costs[agent_id] != 0 else 0.1
        
        # 基于策略梯度范数的重要性
        if hasattr(trainer_list[agent_id], 'last_policy_grad_norm'):
            grad_importance = trainer_list[agent_id].last_policy_grad_norm
        else:
            grad_importance = 1.0
            
        # 综合重要性分数
        importance_scores[agent_id] = reward_importance * 0.4 + cost_importance * 0.4 + grad_importance * 0.2
    
    # 归一化
    if importance_scores.sum() > 0:
        importance_scores = importance_scores / importance_scores.sum()
    else:
        importance_scores = np.ones(num_agents) / num_agents
        
    return importance_scores

def select_agents_by_importance(importance_scores: np.ndarray, 
                               selection_ratio: float = 0.5) -> List[int]:
    """
    基于重要性分数选择智能体
    
    Args:
        importance_scores: 重要性分数
        selection_ratio: 选择比例
    
    Returns:
        selected_agents: 选中的智能体ID列表
    """
    num_agents = len(importance_scores)
    num_selected = max(1, int(num_agents * selection_ratio))
    
    # 基于重要性分数进行概率采样
    selected_agents = np.random.choice(
        num_agents, 
        size=num_selected, 
        replace=False, 
        p=importance_scores
    ).tolist()
    
    return selected_agents

def hierarchical_agent_selection(num_agents: int, 
                                hierarchy_levels: int = 3,
                                current_step: int = 0) -> List[int]:
    """
    分层选择智能体
    
    Args:
        num_agents: 智能体总数
        hierarchy_levels: 层级数
        current_step: 当前步数
    
    Returns:
        selected_agents: 选中的智能体ID列表
    """
    # 根据当前步数确定更新层级
    level = current_step % hierarchy_levels
    
    # 每层选择不同数量的智能体
    if level == 0:  # 高频更新层：选择少量重要智能体
        num_selected = max(1, num_agents // 4)
    elif level == 1:  # 中频更新层：选择中等数量智能体
        num_selected = max(1, num_agents // 2)
    else:  # 低频更新层：选择大部分智能体
        num_selected = max(1, int(num_agents * 0.8))
    
    # 随机选择（可以结合重要性分数改进）
    selected_agents = np.random.choice(
        num_agents, 
        size=num_selected, 
        replace=False
    ).tolist()
    
    return selected_agents

def progressive_agent_selection(num_agents: int, 
                               current_episode: int,
                               total_episodes: int,
                               min_ratio: float = 0.3,
                               max_ratio: float = 1.0) -> List[int]:
    """
    渐进式智能体选择
    
    Args:
        num_agents: 智能体总数
        current_episode: 当前回合
        total_episodes: 总回合数
        min_ratio: 最小选择比例
        max_ratio: 最大选择比例
    
    Returns:
        selected_agents: 选中的智能体ID列表
    """
    # 计算当前选择比例（从min_ratio渐进到max_ratio）
    progress = min(1.0, current_episode / total_episodes)
    current_ratio = min_ratio + (max_ratio - min_ratio) * progress
    
    num_selected = max(1, int(num_agents * current_ratio))
    
    selected_agents = np.random.choice(
        num_agents, 
        size=num_selected, 
        replace=False
    ).tolist()
    
    return selected_agents

def compute_communication_efficiency(selected_agents: List[int], 
                                   total_agents: int) -> float:
    """
    计算通信效率
    
    Args:
        selected_agents: 选中的智能体
        total_agents: 总智能体数
    
    Returns:
        efficiency: 通信效率（0-1之间）
    """
    return len(selected_agents) / total_agents

def adaptive_selection_ratio(performance_history: List[float], 
                           window_size: int = 10,
                           base_ratio: float = 0.5) -> float:
    """
    基于性能历史自适应调整选择比例
    
    Args:
        performance_history: 性能历史记录
        window_size: 窗口大小
        base_ratio: 基础选择比例
    
    Returns:
        adjusted_ratio: 调整后的选择比例
    """
    if len(performance_history) < window_size:
        return base_ratio
    
    # 计算最近窗口的性能趋势
    recent_performance = performance_history[-window_size:]
    performance_trend = np.mean(np.diff(recent_performance))
    
    # 如果性能下降，增加选择比例；如果性能提升，可以减少选择比例
    if performance_trend < 0:  # 性能下降
        adjustment = min(0.2, abs(performance_trend) * 0.1)
        adjusted_ratio = min(1.0, base_ratio + adjustment)
    else:  # 性能提升或稳定
        adjustment = min(0.1, performance_trend * 0.05)
        adjusted_ratio = max(0.2, base_ratio - adjustment)
    
    return adjusted_ratio

class AsyncTrainingMetrics:
    """异步训练指标跟踪"""
    
    def __init__(self):
        self.communication_efficiency_history = []
        self.performance_history = []
        self.selection_history = []
        self.training_time_history = []
    
    def update(self, communication_efficiency: float, 
               performance: float, 
               selected_agents: List[int],
               training_time: float):
        """更新指标"""
        self.communication_efficiency_history.append(communication_efficiency)
        self.performance_history.append(performance)
        self.selection_history.append(selected_agents.copy())
        self.training_time_history.append(training_time)
    
    def get_average_efficiency(self, window_size: int = 100) -> float:
        """获取平均通信效率"""
        if not self.communication_efficiency_history:
            return 0.0
        recent_data = self.communication_efficiency_history[-window_size:]
        return np.mean(recent_data)
    
    def get_performance_trend(self, window_size: int = 50) -> float:
        """获取性能趋势"""
        if len(self.performance_history) < window_size:
            return 0.0
        recent_data = self.performance_history[-window_size:]
        return np.mean(np.diff(recent_data))
    
    def get_training_speedup(self) -> float:
        """计算训练加速比"""
        if len(self.training_time_history) < 2:
            return 1.0
        
        # 假设同步训练时间为选择所有智能体的时间
        sync_time = max(self.training_time_history)
        avg_async_time = np.mean(self.training_time_history)
        
        return sync_time / avg_async_time if avg_async_time > 0 else 1.0